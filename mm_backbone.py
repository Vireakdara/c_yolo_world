# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple
import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig, AutoModelForMaskedLM, AutoTokenizer, CLIPTextModel, RobertaModel)
from transformers import CLIPTextModelWithProjection as CLIPTP
from transformers import AutoTokenizer, AutoModel, BeitConfig, BeitModel, XLMRobertaTokenizer, AlignTextConfig, AlignTextModel
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
class HuggingALIGNLanguageBackbone(BaseModule):
    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 target_hidden_size: int = 512,  # Adjust for downstream task
                 training_use_cache: bool = False) -> None:
        super().__init__()

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        
        # Initialize ALIGN Tokenizer and Model (Assuming ALIGN model supports text input)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AlignTextConfig.from_pretrained(model_name,
                                                      attention_dropout=dropout)
        self.model = AlignTextModel.from_pretrained(        model_name,
            config=self.config,)

        # Fusion layer to combine the embeddings from the model
        self.fusion_layer = nn.Linear(
            self.model.config.hidden_size,
            self.model.config.hidden_size
        )
        self._freeze_modules()

        # Add a projection layer if hidden size needs adjustment
        self.projection = nn.Linear(self.config.hidden_size, target_hidden_size)

        # Optional fusion layer for specific downstream tasks
        self.fusion_layer = nn.Linear(target_hidden_size, target_hidden_size)

        # Freezing specific modules
        self._freeze_modules()

    def forward_tokenizer(self, texts: List[List[str]]):
        """Tokenizes input text using ALIGN tokenizer."""
        flat_text = list(itertools.chain(*texts))
        tokens = self.tokenizer(
            text=flat_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        return tokens.to(self.model.device)

    def forward(self, text: List[List[str]]) -> Tensor:
        """Encodes text using ALIGN and processes embeddings."""
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), "Batch sequence lengths must match"

        tokens = self.forward_tokenizer(text)

        # Forward pass through ALIGN model
        outputs = self.model(**tokens)
        hidden_states = outputs.last_hidden_state

        # Debugging shape
        print(f"Hidden state shape: {hidden_states.shape}")

        # CLS token embeddings and normalization
        cls_embeddings = hidden_states[:, 0, :]  # CLS token
        cls_embeddings = F.normalize(cls_embeddings, p=2, dim=-1)

        # Apply projection layer for dimensionality adjustment
        projected_embeddings = self.projection(cls_embeddings)

        # Debugging shape
        print(f"Projected embeddings shape: {projected_embeddings.shape}")

        # Reshape embeddings back to batch format
        reshaped_embeddings = projected_embeddings.view(-1, num_per_batch[0], projected_embeddings.size(-1))
        return reshaped_embeddings

    def _freeze_modules(self):
        """Freezes specific or all modules of the model."""
        if len(self.frozen_modules) == 0:
            return

        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, param in self.model.named_parameters():
                param.requires_grad = False
            return

        # Freeze specified modules
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        """Ensures frozen modules remain frozen during training."""
        super().train(mode)
        self._freeze_modules()

@MODELS.register_module()
class HuggingQwenLanguageBackbone(BaseModule):
    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache

        # Initialize tokenizer and model specific to Qwen1.5-0.5B
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,  # Ensure hidden states are available
            output_attentions=False,    # Disable attention outputs unless needed
        )

        self._freeze_modules()

    def forward_tokenizer(self, texts):
        """Tokenizes the input texts for the model."""
        flat_text = list(itertools.chain(*texts))
        tokens = self.tokenizer(
            text=flat_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Adjust as per memory constraints
        )
        return tokens.to(device=self.model.device)

    def forward(self, text: List[List[str]]) -> Tensor:
        """Encodes text using Qwen1.5-0.5B and extracts embeddings."""
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            "Number of sequences must be equal in each batch."
        )

        tokens = self.forward_tokenizer(text)

        # Forward pass through the Qwen1.5-0.5B model
        outputs = self.model(**tokens)
        hidden_states = outputs.hidden_states[-1]  # Use the last hidden state

        # Extract CLS token embeddings and normalize
        cls_embeddings = hidden_states[:, 0, :]  # CLS token
        normalized_embeddings = F.normalize(cls_embeddings, p=2, dim=-1)

        # Reshape embeddings back to batch format
        reshaped_embeddings = normalized_embeddings.view(-1, num_per_batch[0], normalized_embeddings.size(-1))
        return reshaped_embeddings

    def _freeze_modules(self):
        """Freezes specific or all modules of the model."""
        if len(self.frozen_modules) == 0:
            return

        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, param in self.model.named_parameters():
                param.requires_grad = False
            return

        # Freeze specific submodules
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        """Ensures frozen modules remain frozen during training."""
        super().train(mode)
        self._freeze_modules()
    
@MODELS.register_module()
class EnhancedTextCLIPBackbone(BaseModule):
    def __init__(self,
                model_name: str,
                frozen_modules: Sequence[str] = (),
                enhanced_text_model_name: str = "D:\\YOLO\\YOLO-World\\roberta-base",
                dropout: float = 0.0,
                training_use_cache: bool = False,
                init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        
        # Initialize CLIP Text Model
        self.clip_tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
        self.clip_text_model = CLIPTextModel.from_pretrained(model_name, config=clip_config)

        # Initialize RoBERTa as Enhanced Text Encoder
        self.enhanced_tokenizer = AutoTokenizer.from_pretrained(enhanced_text_model_name)
        self.enhanced_text_model = RobertaModel.from_pretrained(enhanced_text_model_name)

        # Fusion Layer to combine CLIP and RoBERTa Embeddings
        self.fusion_layer = nn.Linear(
            self.clip_text_model.config.hidden_size + self.enhanced_text_model.config.hidden_size,
            self.clip_text_model.config.hidden_size
        )
        self._freeze_modules()

    def forward_tokenizer(self, texts: List[List[str]]):
        # Tokenize text for both CLIP and RoBERTa
        flat_text = list(itertools.chain(*texts))
        clip_tokens = self.clip_tokenizer(text=flat_text, return_tensors='pt', padding=True)
        enhanced_tokens = self.enhanced_tokenizer(text=flat_text, return_tensors='pt', padding=True)
        return clip_tokens.to(self.clip_text_model.device), enhanced_tokens.to(self.enhanced_text_model.device)

    def forward(self, text: List[List[str]]) -> Tensor:
        # Process text with both CLIP and RoBERTa
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), "Batch sequence lengths must match"
        
        clip_tokens, enhanced_tokens = self.forward_tokenizer(text)
        
        # CLIP Text Embeddings
        clip_outputs = self.clip_text_model(**clip_tokens)
        clip_embeds = clip_outputs.last_hidden_state[:, 0, :]  # CLS token embedding for CLIP
        clip_embeds = clip_embeds / clip_embeds.norm(p=2, dim=-1, keepdim=True)

        # RoBERTa Text Embeddings
        enhanced_outputs = self.enhanced_text_model(**enhanced_tokens)
        enhanced_embeds = enhanced_outputs.last_hidden_state[:, 0, :]  # CLS token embedding for RoBERTa
        enhanced_embeds = enhanced_embeds / enhanced_embeds.norm(p=2, dim=-1, keepdim=True)

        # Concatenate and fuse embeddings
        combined_embeds = torch.cat([clip_embeds, enhanced_embeds], dim=-1)
        fused_embeds = self.fusion_layer(combined_embeds)
        fused_embeds = fused_embeds.reshape(-1, num_per_batch[0], fused_embeds.shape[-1])
        
        return fused_embeds
    
    def _freeze_modules(self):
        if len(self.frozen_modules) == 0:
            # No modules to freeze
            return

        # Freeze all modules if "all" is specified
        if self.frozen_modules[0] == "all":
            # Freeze CLIP text model
            self.clip_text_model.eval()
            for _, module in self.clip_text_model.named_modules():
                for param in module.parameters():
                    param.requires_grad = False

            # Freeze RoBERTa enhanced text model
            self.enhanced_text_model.eval()
            for _, module in self.enhanced_text_model.named_modules():
                for param in module.parameters():
                    param.requires_grad = False
            return

        # Freeze specified modules
        for name, module in self.clip_text_model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

        for name, module in self.enhanced_text_model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

@MODELS.register_module()
class EnhancedTextCLIPBackboneV2(BaseModule):
    def __init__(self,
                model_name: str,
                frozen_modules: Sequence[str] = (),
                enhanced_text_model_name: str = "D:\\YOLO\\YOLO-World\\beit-large-patch16-224",
                dropout: float = 0.0,
                training_use_cache: bool = False,
                init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        
        # Initialize CLIP Text Model
        self.clip_tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
        self.clip_text_model = CLIPTextModel.from_pretrained(model_name, config=clip_config)

        # Initialize BEiT as Enhanced Text Encoder
        self.enhanced_tokenizer = XLMRobertaTokenizer("D:\\YOLO\\YOLO-World\\beit-large-patch16-224\\beit3.spm")
        self.enhanced_text_model = BeitModel.from_pretrained(enhanced_text_model_name)

        # Fusion Layer to combine CLIP and BEiT Embeddings
        self.fusion_layer = nn.Linear(
            self.clip_text_model.config.hidden_size + self.enhanced_text_model.config.hidden_size,
            self.clip_text_model.config.hidden_size
        )
        self._freeze_modules()

    def forward_tokenizer(self, texts: List[List[str]]):
        # Tokenize text for both CLIP and BEiT
        flat_text = list(itertools.chain(*texts))
        clip_tokens = self.clip_tokenizer(text=flat_text, return_tensors='pt', padding=True)
        enhanced_tokens = self.enhanced_tokenizer(text=flat_text, return_tensors='pt', padding=True)
        return clip_tokens.to(self.clip_text_model.device), enhanced_tokens.to(self.enhanced_text_model.device)

    def forward(self, text: List[List[str]]) -> torch.Tensor:
        # Process text with both CLIP and BEiT
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), "Batch sequence lengths must match"
        
        clip_tokens, enhanced_tokens = self.forward_tokenizer(text)
        
        # CLIP Text Embeddings
        clip_outputs = self.clip_text_model(**clip_tokens)
        clip_embeds = clip_outputs.last_hidden_state[:, 0, :]  # CLS token embedding for CLIP
        clip_embeds = clip_embeds / clip_embeds.norm(p=2, dim=-1, keepdim=True)

        # BEiT Text Embeddings
        enhanced_outputs = self.enhanced_text_model(**enhanced_tokens)
        enhanced_embeds = enhanced_outputs.last_hidden_state[:, 0, :]  # CLS token embedding for BEiT
        enhanced_embeds = enhanced_embeds / enhanced_embeds.norm(p=2, dim=-1, keepdim=True)

        # Concatenate and fuse embeddings
        combined_embeds = torch.cat([clip_embeds, enhanced_embeds], dim=-1)
        fused_embeds = self.fusion_layer(combined_embeds)
        fused_embeds = fused_embeds.reshape(-1, num_per_batch[0], fused_embeds.shape[-1])
        
        return fused_embeds
    
    def _freeze_modules(self):
        if len(self.frozen_modules) == 0:
            # No modules to freeze
            return
        
        if "all" in self.frozen_modules:
            # Freeze all parameters
            for param in self.clip_text_model.parameters():
                param.requires_grad = False
            for param in self.enhanced_text_model.parameters():
                param.requires_grad = False
        else:
            # Freeze specific submodules
            for name, module in self.clip_text_model.named_modules():
                if any(name.startswith(fm) for fm in self.frozen_modules):
                    for param in module.parameters():
                        param.requires_grad = False
            for name, module in self.enhanced_text_model.named_modules():
                if any(name.startswith(fm) for fm in self.frozen_modules):
                    for param in module.parameters():
                        param.requires_grad = False

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
        
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the SBERT model
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set dropout if supported by the model's configuration
        if hasattr(self.model.config, 'attention_probs_dropout_prob'):
            self.model.config.attention_probs_dropout_prob = dropout
        if hasattr(self.model.config, 'hidden_dropout_prob'):
            self.model.config.hidden_dropout_prob = dropout
        
        # Use the model's hidden size to set the dimensions for the fusion layer
        hidden_size = self.model.config.hidden_size
        self.fusion_layer = nn.Linear(hidden_size, hidden_size)
        
        # Freeze specified modules if necessary
        self._freeze_modules()

    def forward_tokenizer(self, texts: List[List[str]]):
        """Tokenizes input text."""
        flat_text = list(itertools.chain(*texts))
        tokens = self.tokenizer(
            text=flat_text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512  # Ensure compatibility with model constraints
        )
        return tokens.to(self.model.device)

    def forward(self, text: List[List[str]]) -> torch.Tensor:
        """Processes the text and outputs normalized embeddings."""
        # Ensure equal sequence lengths in the batch
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            "Number of sequences per batch must be equal"
        )
        
        # Tokenize the input text
        tokens = self.forward_tokenizer(text)
        
        # Pass tokens through the model to get outputs
        outputs = self.model(**tokens)
        
        # Use mean pooling for sentence embeddings
        hidden_states = outputs.last_hidden_state
        txt_feats = hidden_states.mean(dim=1)
        
        # Normalize the embeddings
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        
        # Pass through the fusion layer (if necessary)
        txt_feats = self.fusion_layer(txt_feats)
        
        # Reshape to match input batch structure
        txt_feats = txt_feats.view(-1, num_per_batch[0], txt_feats.size(-1))
        return txt_feats

    def _freeze_modules(self):
        """Freezes specified modules of the model."""
        if len(self.frozen_modules) == 0:
            return
        
        if self.frozen_modules[0] == "all":
            for param in self.model.parameters():
                param.requires_grad = False
            return

        # Freeze specific modules
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        """Ensures frozen modules remain frozen during training."""
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
        print("Text >>>>>", text)
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
