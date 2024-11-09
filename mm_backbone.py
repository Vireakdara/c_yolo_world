# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple
import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig, AutoModelForMaskedLM)
from transformers import CLIPTextModelWithProjection as CLIPTP
from transformers import AutoTokenizer, AutoModel, BeitConfig, BeitModel, XLMRobertaTokenizer
from typing import List, Sequence


@MODELS.register_module()
class HuggingVisionBackbone(BaseModule):

    def __init__(self,
                 model_name: str,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 norm_eval: bool = True,
                 frozen_modules: Sequence[str] = (),
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.model = AutoModel.from_pretrained(model_name)

        self._freeze_modules()

    def forward(self, image: Tensor) -> Tuple[Tensor]:
        encoded_dict = self.image_model(pixel_values=image,
                                        output_hidden_states=True)
        hidden_states = encoded_dict.hidden_states
        img_feats = encoded_dict.get('reshaped_hidden_states', hidden_states)
        img_feats = [img_feats[i] for i in self.image_out_indices]
        return tuple(img_feats)

    def _freeze_modules(self):
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):

    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name,
                                                     attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pt', padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        # print("Text1 >>>>>>>>", text)

        text = text.to(device=self.model.device)
        # print("Text2 >>>>>>>>", text)
        txt_outputs = self.model(**text)
        return;
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                      txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


@MODELS.register_module()
class PseudoLanguageBackbone(BaseModule):
    """Pseudo Language Backbone
    Args:
        text_embed_path (str): path to the text embedding file
    """

    def __init__(self,
                 text_embed_path: str = "",
                 test_embed_path: str = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        # {text:embed}
        self.text_embed = torch.load(text_embed_path, map_location='cpu')
        if test_embed_path is None:
            self.test_embed = self.text_embed
        else:
            self.test_embed = torch.load(test_embed_path)
        self.register_buffer("buff", torch.zeros([
            1,
        ]))

    def forward_cache(self, text: List[List[str]]) -> Tensor:
        if not hasattr(self, "cache"):
            self.cache = self.forward_text(text)
        return self.cache

    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        else:
            return self.forward_cache(text)

    def forward_text(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        if self.training:
            text_embed_dict = self.text_embed
        else:
            text_embed_dict = self.test_embed
        text_embeds = torch.stack(
            [text_embed_dict[x.split("/")[0]] for x in text])
        # requires no grad and force to float
        text_embeds = text_embeds.to(
            self.buff.device).requires_grad_(False).float()
        text_embeds = text_embeds.reshape(-1, num_per_batch[0],
                                          text_embeds.shape[-1])
        return text_embeds


@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):

    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
        if self.with_text_model:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)

## SBERT MODEL TEST IN PROGRESS
@MODELS.register_module()
class HuggingSBERTLanguageBackbone(nn.Module):

    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False) -> None:
        super().__init__()
        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the SBERT model
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set dropout if required (might be model-specific)
        if hasattr(self.model.config, 'attention_probs_dropout_prob'):
            self.model.config.attention_probs_dropout_prob = dropout
        if hasattr(self.model.config, 'hidden_dropout_prob'):
            self.model.config.hidden_dropout_prob = dropout

        # Freeze specified modules
        self._freeze_modules()

    def forward_tokenizer(self, texts: List[str]):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pt', padding=True, truncation=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        # Check for consistent number of sentences per batch
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        
        # Tokenize and encode
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True, truncation=True)
        text = text.to(device=self.model.device)
        
        # Get embeddings from SBERT model
        outputs = self.model(**text)
        txt_feats = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling for embeddings
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)  # Normalize embeddings
        
        # Reshape to match input batch structure
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):
        if len(self.frozen_modules) == 0:
            return
        
        if self.frozen_modules[0] == "all":
            for param in self.model.parameters():
                param.requires_grad = False
            return

        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()

## BEit3 MODEL TEST IN PROGRESS
@MODELS.register_module()
class HuggingBEiT3LanguageBackbone(nn.Module):
    def __init__(self,
                 model_name: str = 'microsoft/beit-large-patch16-224', 
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False) -> None:
        super().__init__()
        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load BEiT-3 model in "text-only" mode
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # Apply dropout settings if necessary (BEiT-3 may use specific configuration parameters)
        if hasattr(self.model.config, 'attention_probs_dropout_prob'):
            self.model.config.attention_probs_dropout_prob = dropout
        if hasattr(self.model.config, 'hidden_dropout_prob'):
            self.model.config.hidden_dropout_prob = dropout

        # Freeze specified layers if needed
        self._freeze_modules()

    def forward_tokenizer(self, texts: List[str]):
        # Tokenizes input texts and stores on the correct device
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pt', padding=True, truncation=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        # Check consistency in batch dimensions
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')

        # Flatten, tokenize, and move to device
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True, truncation=True)
        text = text.to(device=self.model.device)
        
        # Forward pass through BEiT-3 text encoder
        outputs = self.model(**text)
        # Get embeddings from the final layer, apply pooling to get sentence embeddings
        txt_feats = outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence-level embeddings
        
        # Normalize embeddings for contrastive tasks
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        
        # Reshape to match input batch structure
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):
        # Freezes specified layers of the BEiT-3 model
        if len(self.frozen_modules) == 0:
            return
        
        if self.frozen_modules[0] == "all":
            for param in self.model.parameters():
                param.requires_grad = False
            return

        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()

## HuggingBeitImageBackbone MODEL TEST IN PROGRESS
@MODELS.register_module()
class HuggingBeitImageBackbone(nn.Module):
    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False) -> None:
        super().__init__()
        
        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = XLMRobertaTokenizer("D:\\YOLO\\YOLO-World\\beit-large-patch16-224\\beit3.spm")
        
        # Set up the BEiT configuration with dropout adjustments
        beit_config = BeitConfig.from_pretrained(model_name)
        beit_config.hidden_dropout_prob = dropout
        self.model = BeitModel.from_pretrained(model_name, config=beit_config)
        
        # Apply dropout settings if necessary
        if hasattr(self.model.config, 'attention_probs_dropout_prob'):
            self.model.config.attention_probs_dropout_prob = dropout
        if hasattr(self.model.config, 'hidden_dropout_prob'):
            self.model.config.hidden_dropout_prob = dropout

        # Freeze specified layers if needed
        self._freeze_modules()

    def forward_tokenizer(self, texts: List[str]):
        # Tokenizes input texts; cache based on training_use_cache flag
        if self.training_use_cache and not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pt', padding=True, truncation=True)
            self.text = text.to(device=self.model.device)
        return self.text if self.training_use_cache else self.tokenizer(text=texts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
   
    def forward(self, text: List[List[str]]) -> Tensor:
        # Check consistency in batch dimensions
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), 'Inconsistent batch sizes'
        
        # Flatten, tokenize, and move to device
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        text = text.to(device=self.model.device)
        # Forward pass through BEiT text encoder
        # print("Text >>>>>", text)
        outputs = self.model(**text)
  
        # Get embeddings, apply pooling
        txt_feats = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        # Normalize embeddings for contrastive tasks
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        
        # Reshape to match input batch structure
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):
        # Freezes specified layers of the BEiT model
        if len(self.frozen_modules) == 0:
            return
        
        if self.frozen_modules[0] == "all":
            for param in self.model.parameters():
                param.requires_grad = False
            return

        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()  # Re-apply module freezing on train toggle
