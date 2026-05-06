from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM

from llava.model.llava_arch_gemma import (
    LlavaMetaModel,
    LlavaMetaForCausalLM,
    build_ecg_projector,
)
from llava.model.ecg_encoder.extract_ecg_feature import ECGEncoder

class LlavaGemmaConfig(AutoConfig):
    model_type = "llava_gemma"


class LlavaGemmaModel(LlavaMetaModel, AutoModelForCausalLM):
    config_class = AutoConfig
    
    def __init__(self, config: AutoConfig):
        super().__init__(config)


class LlavaGemmaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM, LlavaMetaModel):
    config_class = LlavaGemmaConfig
    supports_gradient_checkpointing = True

    def patch_gemma3_config(self, config):
        """
        Promote selected fields from Gemma3Config.text_config / .decoder to the
        top-level config so LlamaForCausalLM.__init__ can read them directly.

        Read-only: text_config / decoder keep their original PretrainedConfig
        type, so transformers' GenerationConfig.from_model_config() -- which
        calls decoder_config.to_dict() -- keeps working without site-packages
        patches.
        """
        def _read(src, key):
            if src is None:
                return (None, False)
            if isinstance(src, dict):
                return (src.get(key), key in src)
            return (getattr(src, key, None), hasattr(src, key))

        text_cfg = getattr(config, "text_config", None)
        decoder_cfg = getattr(config, "decoder", None)

        promote_keys = [
            # Basic structure
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_position_embeddings",
            # Numerical stability & normalization
            "rms_norm_eps",
            "torch_dtype",
            # Dropout / bias
            "attention_dropout",
            "attention_bias",
            # RoPE related
            "rope_scaling",
            "rope_theta",
            # Sliding window related
            "sliding_window",
            "sliding_window_pattern",
        ]

        for k in promote_keys:
            if hasattr(config, k):
                continue
            for src in (text_cfg, decoder_cfg):
                v, found = _read(src, k)
                if found:
                    setattr(config, k, v)
                    break

        if not hasattr(config, "mlp_bias"):
            config.mlp_bias = True

        if not hasattr(config, "hidden_act"):
            v, found = _read(text_cfg, "hidden_activation")
            if found:
                config.hidden_act = v

        config.use_cache = True
        return config



    def __init__(self, config):
        self.config = self.patch_gemma3_config(config)
        super().__init__(self.config)

        self.ecg_tower = ECGEncoder()
        self.mm_projector = build_ecg_projector()

        self.post_init()

    def get_model(self):
        return self.model

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, AutoModelForCausalLM):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ecgs=None
    ):
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            ecgs
        )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

AutoConfig.register("llava_gemma", LlavaGemmaConfig)
AutoModelForCausalLM.register(LlavaGemmaConfig, LlavaGemmaForCausalLM)


if __name__ == "__main__":
    import os
    medgemma_path = os.environ.get("UNIPACT_MEDGEMMA_PATH")
    if not medgemma_path:
        raise SystemExit(
            "Set UNIPACT_MEDGEMMA_PATH to a local copy of the MedGemma "
            "checkpoint to run this smoke test."
        )
    config = LlavaGemmaConfig.from_pretrained(medgemma_path)
    model = LlavaGemmaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=medgemma_path, config=config
    )
    print(model.config.text_config)