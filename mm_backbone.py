# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple
import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
import torch.nn.functional as F
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig, AutoModelForMaskedLM, AutoTokenizer, CLIPTextModel, RobertaModel)
from transformers import CLIPTextModelWithProjection as CLIPTP
from transformers import AutoTokenizer, AutoModel, BeitConfig, BeitModel, XLMRobertaTokenizer, AlignTextConfig, AlignTextModel, AltCLIPTextModel , AltCLIPTextConfig
from typing import List, Sequence, Dict



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
class HuggingAltCLIPLanguageBackboneWithPrompts(BaseModule):
    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 target_hidden_size: int = 512,  # For downstream tasks
                 prompt_length: int = 5,  # Number of soft prompt tokens
                 training_use_cache: bool = False) -> None:
        super().__init__()

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AltCLIPTextConfig.from_pretrained(
            model_name,
            attention_dropout=dropout
        )
        self.config.hidden_size = 768  # Explicitly set hidden size if necessary
        self.model = AltCLIPTextModel.from_pretrained(model_name, config=self.config)

        # Add a trainable soft prompt embedding layer
        self.soft_prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, self.config.hidden_size)  # Initialize randomly
        )

        # Add a projection layer for dimensionality adjustment
        self.projection = nn.Linear(self.config.hidden_size, target_hidden_size)

        # Optional fusion layer for specific downstream tasks
        self.fusion_layer = nn.Linear(target_hidden_size, target_hidden_size)

        # Apply module freezing if needed
        self._freeze_modules()

    def forward_tokenizer(self, texts: List[List[str]]) -> Dict[str, Tensor]:
        """Tokenizes input texts and prepends soft prompts."""
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
        tokens = self.forward_tokenizer(text)

        # Extract input embeddings
        input_embeddings = self.model.get_input_embeddings()(tokens.input_ids)  # Shape: [batch_size, seq_len, hidden_size]
        print(f"Input embeddings shape: {input_embeddings.shape}")

        # Add soft prompts to the input embeddings
        batch_size = input_embeddings.size(0)
        soft_prompts = self.soft_prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, prompt_len, hidden_size]
        print(f"Soft prompts shape: {soft_prompts.shape}")
        
        input_embeddings = torch.cat([soft_prompts, input_embeddings], dim=1)  # Concatenate prompts
        print(f"Input embeddings after concatenation: {input_embeddings.shape}")

        # Adjust attention mask for the added prompts
        attention_mask = tokens.attention_mask
        prompt_mask = torch.ones(batch_size, self.soft_prompt_embeddings.size(0), device=attention_mask.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        print(f"Attention mask shape: {attention_mask.shape}")

        # Forward pass through the AltCLIP model
        outputs = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len + prompt_len, hidden_size]
        print(f"Hidden states shape: {hidden_states.shape}")

        # Use CLS token embeddings
        cls_embeddings = hidden_states[:, 0, :]  # Shape: [batch_size, hidden_size]
        print(f"CLS embeddings shape: {cls_embeddings.shape}")

        # Normalize embeddings
        cls_embeddings = F.normalize(cls_embeddings, p=2, dim=-1)

        # Apply projection layer
        projected_embeddings = self.projection(cls_embeddings)  # Shape: [batch_size, target_hidden_size]
        print(f"Projected embeddings shape: {projected_embeddings.shape}")

        return projected_embeddings
    
    def _freeze_modules(self):
        """Freezes specific or all modules of the model."""
        if not self.frozen_modules:
            return  # No modules to freeze

        if "all" in self.frozen_modules:
            for param in self.model.parameters():
                param.requires_grad = False
            return

        # Freeze specified modules
        for name, module in self.model.named_modules():
            if any(name.startswith(fm) for fm in self.frozen_modules):
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Ensures frozen modules remain frozen during training."""
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

        # Correctly initialize the fusion layer to match the SBERT output size
        self.hidden_size = self.model.config.hidden_size
        self.fusion_layer = nn.Linear(self.hidden_size, 256)  # Match downstream expectations

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

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def forward(self, text: List[List[str]]) -> torch.Tensor:
        """Processes the text and outputs normalized embeddings."""
        # Ensure equal sequence lengths in the batch
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), "Number of sequences per batch must be equal"

        # Tokenize the input text
        tokens = self.forward_tokenizer(text)

        # Pass tokens through the model to get outputs
        with torch.no_grad():
            model_output = self.model(**tokens)

        # Debugging: Check hidden states
        print(f"Shape of token embeddings (model_output[0]): {model_output[0].shape}")  # Should be [batch_size, seq_len, hidden_size]

        # Perform mean pooling
        sentence_embeddings = self.mean_pooling(model_output, tokens['attention_mask'])
        print(f"Shape after pooling: {sentence_embeddings.shape}")  # Should be [batch_size, hidden_size]

        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        print(f"Shape after normalization: {sentence_embeddings.shape}")  # Should be [batch_size, hidden_size]

        # Pass through the fusion layer
        fused_embeddings = self.fusion_layer(sentence_embeddings)
        print(f"Shape after fusion layer: {fused_embeddings.shape}")  # Should be [batch_size, 256]

        # Reshape for compatibility if needed
        reshaped_feats = fused_embeddings.view(-1, num_per_batch[0], fused_embeddings.size(-1))
        print(f"Shape after reshaping: {reshaped_feats.shape}")  # Should be [batch_size, num_sequences, 256]

        # Flatten reshaped_feats
        flat_feats = reshaped_feats.view(-1, fused_embeddings.size(-1))
        print(f"Shape after flattening: {flat_feats.shape}")  # [total_sequences, 256]

        # Ensure compatibility with downstream requirements
        required_size = 512  # Adjust based on the downstream model
        if flat_feats.size(0) > required_size:
            flat_feats = flat_feats[:required_size]  # Select the first `required_size` sequences
        elif flat_feats.size(0) < required_size:
            padding = torch.zeros((required_size - flat_feats.size(0), flat_feats.size(1)),
                                device=flat_feats.device)
            flat_feats = torch.cat([flat_feats, padding], dim=0)

        # After ensuring `flat_feats` is `[512, 256]`
        print(f"Final shapes for downstream: {flat_feats.shape}")  # Should be [512, 256]

        # Downstream layer compatibility
        # If downstream expects [256, 512], transpose the input
        flat_feats = flat_feats.T  # Transpose to [256, 512] if required
        print(f"Final shape after transpose: {flat_feats.shape}")  # Verify dimensions
        return flat_feats

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
