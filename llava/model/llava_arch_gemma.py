from abc import ABC, abstractmethod
import os
from .ecg_encoder.extract_ecg_feature import ECGEncoder
from .multimodal_projector.builder import build_ecg_projector
from llava.constants import IGNORE_INDEX, ECG_TOKEN_INDEX
import torch

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.ecg_tower = ECGEncoder()
        self.mm_projector = build_ecg_projector(config)

    def get_ecg_tower(self):
        ecg_tower = getattr(self, 'ecg_tower', None)
        if type(ecg_tower) is list:
            ecg_tower = ecg_tower[0]
        return ecg_tower

    def initialize_ecg_modules(self, model_args, fsdp=None):

        if self.get_ecg_tower() is None:
            ecg_tower = ECGEncoder()

            if fsdp is not None and len(fsdp) > 0:
                self.ecg_tower = [ecg_tower]
            else:
                self.ecg_tower = ecg_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                ecg_tower = self.ecg_tower[0]
            else:
                ecg_tower = self.ecg_tower
            ecg_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = ecg_tower.hidden_size

        if getattr(self, 'ecg_projector', None) is None:
            self.ecg_projector = build_ecg_projector(self.config)

        else:
            for p in self.ecg_projector.parameters():
                p.requires_grad = True

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_ecg_tower(self):
        return self.ecg_tower

    def encode_ecgs(self, ecgs):
        ecg_tower = self.ecg_tower
        checkpoint_path = getattr(self.config, 'ecg_encoder_dir', None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_state = checkpoint['model']
            
            model_state_dict = dict(ecg_tower.named_parameters())
            
            for model_key in model_state_dict:
                checkpoint_key = model_key.replace('model.', 'ecg_encoder.')
                if checkpoint_key in checkpoint_state:
                    with torch.no_grad():
                        param = model_state_dict[model_key]
                        param.copy_(checkpoint_state[checkpoint_key])
        
        x = ecgs
        model = ecg_tower.model if hasattr(ecg_tower, 'model') else ecg_tower

        if hasattr(model, 'feature_extractor'):
            feature_extractor = model.feature_extractor
            if hasattr(feature_extractor, 'conv_layers'):
                if x.shape[1] > x.shape[2]:
                    x = x.permute(0, 2, 1)

        ecg_features = ecg_tower(ecgs)
        ecg_features = self.mm_projector(ecg_features)
        
        return ecg_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        ecgs, ecg_sizes=None
    ):
        ecg_tower = self.ecg_tower
        if ecg_tower is None or ecgs is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        ecg_features = self.encode_ecgs(ecgs)
        
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_ecg_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_ecgs = (cur_input_ids == ECG_TOKEN_INDEX).sum()
            if num_ecgs == 0:
                return _input_ids, _position_ids, _attention_mask, past_key_values, None, _labels

            ecg_token_indices = [-1] + torch.where(cur_input_ids == ECG_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(ecg_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[ecg_token_indices[i]+1:ecg_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[ecg_token_indices[i]+1:ecg_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            
            cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_ecgs + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_ecgs:
                    cur_ecg_features = ecg_features[cur_ecg_idx].unsqueeze(0)  # Add batch dimension
                    cur_ecg_idx += 1
                    cur_new_input_embeds.append(cur_ecg_features)
                    cur_new_labels.append(torch.full((cur_ecg_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
