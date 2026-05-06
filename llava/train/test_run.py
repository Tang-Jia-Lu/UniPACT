import copy
import csv
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.data import Dataset
import transformers

mp.set_start_method('spawn', force=True)

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from llava.constants import IGNORE_INDEX, DEFAULT_ECG_TOKEN, DEFAULT_ECG_START_TOKEN, DEFAULT_ECG_END_TOKEN
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_ecg_token
from llava.model.ecg_encoder.extract_ecg_feature import ECGDataset
from llava.model.language_model.llava_gemma import LlavaGemmaForCausalLM

from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import ast


def calculate_auc_accuracy_to_txt(
    input_csv_path: str,
    output_txt_path: str
):
    """Calculate per-group AUC and Accuracy and save a summary to a TXT file."""

    df = pd.read_csv(input_csv_path, on_bad_lines='skip')

    def safe_literal_eval(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return None

    df['group_key_tuple'] = df['group_key'].apply(safe_literal_eval)
    df['group_main'] = df['group_key_tuple'].apply(
        lambda x: x[0] if isinstance(x, (list, tuple)) else x
    )

    df['group_label'] = df['group_main'].apply(
        lambda x: 'safe' if x == 'safe' else x
    )

    results = []

    for group_name, group in df.groupby('group_label'):
        probs = group[['yes_prob', 'no_prob']].astype(float).values
        y_pred = probs[:, 1]
        y_true = [0 if gt == 'Yes' else 1 for gt in group['GT']]

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan

        y_pred_label = ['Yes' if p[0] >= p[1] else 'No' for p in probs]
        accuracy = accuracy_score(group['GT'], y_pred_label)

        results.append({
            'group': group_name,
            'AUC': auc,
            'Accuracy': accuracy,
            'n_samples': len(group)
        })

    results_df = pd.DataFrame(results)

    lines = []
    lines.append("=== Detailed Results by Group ===\n")
    lines.append(results_df.to_string(index=False))
    lines.append("\n\n")

    mean_auc = results_df['AUC'].mean()
    mean_acc = results_df['Accuracy'].mean()
    sum_samples = results_df['n_samples'].sum()

    lines.append("=== Overall Statistics ===\n")
    lines.append(f"Mean AUC: {mean_auc:.4f}\n")
    lines.append(f"Mean Accuracy: {mean_acc:.4f}\n")
    lines.append(f"Total Samples: {sum_samples}\n\n")

    mortality_df = results_df[results_df['group'].str.contains('mortality', na=False)]
    icu_df = results_df[results_df['group'].str.contains('icu', na=False)]
    diagnose_df = results_df[results_df['group'].str.contains('diagnose', na=False)]
    other_df = results_df[
        ~(results_df['group'].str.contains('mortality|icu|diagnose', na=False))
    ]

    def group_stats_to_lines(df, name):
        l = []
        mean_auc = df['AUC'].mean()
        mean_acc = df['Accuracy'].mean()
        sum_samples = df['n_samples'].sum()
        auc_over_08 = (df['AUC'] > 0.8).sum()
        ratio = auc_over_08 / len(df) if len(df) > 0 else 0

        l.append(f"=== {name} Statistics ===\n")
        l.append(f"Mean AUC: {mean_auc:.4f}\n")
        l.append(f"Mean Accuracy: {mean_acc:.4f}\n")
        l.append(f"Total Samples: {sum_samples}\n")
        l.append(f"Groups with AUC > 0.8: {auc_over_08}\n")
        l.append(f"Ratio of AUC > 0.8: {ratio:.4f}\n\n")
        return l

    lines.append("=" * 60 + "\n")
    lines.append("Detailed Statistics by Task Type\n")
    lines.append("=" * 60 + "\n\n")

    lines.append("[Mortality Prediction Task Details]\n")
    lines.append(mortality_df.to_string(index=False) + "\n")
    lines.extend(group_stats_to_lines(mortality_df, "Mortality Prediction"))

    lines.append("[ICU-Related Task Details]\n")
    lines.append(icu_df.to_string(index=False) + "\n")
    lines.extend(group_stats_to_lines(icu_df, "ICU-Related"))

    lines.append("[Diagnosis Task Details]\n")
    lines.append(diagnose_df.to_string(index=False) + "\n")
    lines.extend(group_stats_to_lines(diagnose_df, "Diagnosis"))

    lines.append("[Other Task Details]\n")
    lines.append(other_df.to_string(index=False) + "\n")
    lines.extend(group_stats_to_lines(other_df, "Other"))

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    ecg_tower: Optional[str] = field(default=None)
    ecg_encoder_dir: Optional[str] = field(default=None)
    mm_ecg_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_ecg_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    ecg_data_path: str = field(default="",
                              metadata={"help": "Path to the local MIMIC-IV-ECG WFDB root."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantification."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    tune_1_mm_mlp_adapter: bool = field(default=False)
    tune_2_lora: bool = field(default=False)
    seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
    output_dir: str = field(default="")
    output_csv_dir: str = field(default="")


def find_all_linear_names(model, training_args):
    cls = torch.nn.Linear
    lora_module_names = set()
    if "_W_" in training_args.output_dir:
        multimodal_keywords_skip = ['mm_projector']
    elif "_WO_" in training_args.output_dir:
        multimodal_keywords_skip = ['mm_projector', 'ecg_tower']
    else:
        multimodal_keywords_skip = ['mm_projector', 'ecg_tower']
    for name, module in model.named_modules():
        if name == 'lm_head':
            continue
        if any(mm_keyword in name for mm_keyword in multimodal_keywords_skip):
            continue
        elif isinstance(module, cls):
            lora_module_names.add(name)
    return lora_module_names


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_ECG_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_ECG_TOKEN, '').strip()
                sentence['value'] = DEFAULT_ECG_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_ECG_TOKEN, '<ECG>' + DEFAULT_ECG_TOKEN + '</ECG>')

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_ecg: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_ecg:
        input_ids = torch.stack([tokenizer_ecg_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_ecg:
                round_len = len(tokenizer_ecg_token(rou, tokenizer))
                instruction_len = len(tokenizer_ecg_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_ecg: bool = False
) -> Dict:
    return preprocess_v1(sources, tokenizer, has_ecg=has_ecg)


class LazySupervisedECGDataset(Dataset):
    """Dataset for supervised fine-tuning with ECG data."""

    def _group_by_task_and_answer(self, data: List[Dict]) -> Dict:
        grouped = {}
        for item in data:
            task_id = item.get("id", "")
            task = "_".join(task_id.split("_")[:-1])
            conversations = item.get("conversations", [])
            for conv in conversations:
                if conv.get("from") == "gpt":
                    answer = conv.get("value", "").strip()
                    key = (task, answer)
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(item)
        return grouped

    def _sample_grouped(self, grouped: Dict) -> List:
        random.seed(self.seed)
        sampled = []
        for key, group in grouped.items():
            sampled.extend(random.sample(group, min(self.n_sample, len(group))))
        return sampled

    def resample(self, seed=None, epoch=None):
        if seed is not None:
            self.seed = seed
        if epoch is not None:
            self.epoch = epoch
        self.list_data_dict = self._sample_grouped(self.grouped)

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 initial_seed: int = 42):
        super(LazySupervisedECGDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.n_sample = 5000
        self.seed = initial_seed
        self.grouped = self._group_by_task_and_answer(list_data_dict)
        self.list_data_dict = self._sample_grouped(self.grouped)

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.ecg_dataset = ECGDataset(data_args.ecg_data_path)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            ecg_tokens = 128 if 'ecg' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + ecg_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'ecg' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        item = self.list_data_dict[i]
        task_id = item.get("id", "")
        task = "_".join(task_id.split("_")[:-1])
        conversations = item.get("conversations", [])
        answer = ""
        for conv in conversations:
            if conv.get("from") == "gpt":
                answer = conv.get("value", "").strip()
                break
        group_key = (task, answer)

        if 'ecg' in sources[0]:
            ecg_file = self.list_data_dict[i]['ecg']
            ecg_tensor = self.ecg_dataset.load_ecg_tensor(ecg_file)
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_ecg=('ecg' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if 'ecg' in self.list_data_dict[i]:
            data_dict['ecg'] = ecg_tensor

        data_dict['group_key'] = group_key

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'ecg' in instances[0]:
            ecgs = [instance['ecg'] for instance in instances]
            if all(x is not None and x.shape == ecgs[0].shape for x in ecgs):
                batch['ecgs'] = torch.stack(ecgs)
            else:
                batch['ecgs'] = ecgs

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, initial_seed: int = 42) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    ann_list = data_args.data_path.split('::')
    train_dataset = LazySupervisedECGDataset(tokenizer=tokenizer,
                                            data_path=ann_list[0],
                                            data_args=data_args,
                                            initial_seed=initial_seed)
    val_dataset = LazySupervisedECGDataset(tokenizer=tokenizer,
                                            data_path=ann_list[1],
                                            data_args=data_args,
                                            initial_seed=initial_seed + 1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)


def test(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    if model_args.ecg_tower is not None:

        model = LlavaGemmaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )

        for key in list(vars(model.config).keys()):
            value = getattr(model.config, key)
            if isinstance(value, dict):
                delattr(model.config, key)

        ecg_encoder_checkpoint_path = model_args.ecg_encoder_dir

        checkpoint = torch.load(ecg_encoder_checkpoint_path, map_location='cuda:0')
        model_data = checkpoint['model']
        new_model_data = {}

        special_keys = {
            "multi_modal_ecg_proj.weight": "proj.weight",
            "multi_modal_ecg_proj.bias": "proj.bias",
            "multi_modal_ecg_pooler.dense.weight": "pooler.dense.weight",
            "multi_modal_ecg_pooler.dense.bias": "pooler.dense.bias"
        }

        for k, v in model_data.items():
            if k in special_keys:
                new_key = special_keys[k]
                new_model_data[new_key] = v
            elif k.startswith("model."):
                new_key = "ecg_encoder." + k[len("model."):]
                new_model_data[new_key] = v
            else:
                new_model_data[k] = v

        checkpoint['model'] = new_model_data

        model.ecg_tower.load_state_dict(checkpoint['model'], strict=False)

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, training_args),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.ecg_tower is not None:
        ecg_tower = model.ecg_tower

        ecg_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.is_multimodal = True

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              initial_seed=training_args.seed)

    if 'eval_dataset' in data_module and data_module['eval_dataset'] is not None:
        eval_dataset = data_module['eval_dataset']

        with torch.inference_mode():

            import glob
            model_pattern = f"{training_args.output_dir}/checkpoint-*/full_model.bin"
            model_files = glob.glob(model_pattern)
            if not model_files:
                raise FileNotFoundError(f"No model files found matching pattern: {model_pattern}")
            file_path = model_files[0]

            state_dict = torch.load(file_path, map_location="cuda")

            # Bridge ecg_tower key mismatch (plain Linear in saved ckpt vs PEFT-wrapped
            # Linear in current model): rename .weight/.bias -> .base_layer.weight/.bias
            # and inject default lora_A/B from the live model.
            model_sd = model.state_dict()
            remapped = {}
            for k, v in state_dict.items():
                if k not in model_sd and (k.endswith(".weight") or k.endswith(".bias")):
                    prefix, suffix = k.rsplit(".", 1)
                    candidate = f"{prefix}.base_layer.{suffix}"
                    if candidate in model_sd:
                        remapped[candidate] = v
                        continue
                remapped[k] = v
            for k, v in model_sd.items():
                if k not in remapped and (".lora_A." in k or ".lora_B." in k):
                    remapped[k] = v.detach().clone()
            state_dict = remapped

            model.load_state_dict(state_dict, strict=True)

            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            total_samples = len(eval_dataset)

            csv_file = os.path.join(training_args.output_dir, f"{training_args.output_csv_dir}_results.csv")
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)

            for i in range(total_samples):
                sample0 = eval_dataset[i]
                sample = {
                    'input_ids': sample0['input_ids'][:-2],
                    'labels': sample0['labels'][:-2],
                    **{k: v for k, v in sample0.items() if k not in ['input_ids', 'labels']}
                }

                input_ids = sample['input_ids'].unsqueeze(0).to(device)
                group_key = sample['group_key']

                ecgs = sample['ecg'].unsqueeze(0)
                if training_args.bf16:
                    ecgs = ecgs.to(dtype=torch.bfloat16, device=device)
                elif training_args.fp16:
                    ecgs = ecgs.to(dtype=torch.float16, device=device)
                else:
                    ecgs = ecgs.to(dtype=torch.float32, device=device)

                try:
                    outputs = model(
                        input_ids=input_ids,
                        ecgs=ecgs,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=False,
                        return_dict=True
                    )

                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    last_token_probs = probs[:, -1, :]
                    topk_probs, topk_indices = torch.topk(last_token_probs, k=1, dim=-1)
                    for idx, (prob, token_id) in enumerate(zip(topk_probs[0], topk_indices[0])):
                        token_str = tokenizer.decode([token_id.item()])

                    yes_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
                    no_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]
                    yes_prob = last_token_probs[0, yes_token_id].item()
                    no_prob = last_token_probs[0, no_token_id].item()

                    row = {
                        "index": i + 1,
                        "total": total_samples,
                        "yes_prob": yes_prob,
                        "no_prob": no_prob,
                        "output": token_str,
                        "group_key": str(group_key),
                        "GT": group_key[1]
                    }
                    write_header = (i == 0)
                    with open(csv_file, "a", newline='', encoding="utf-8-sig") as f:
                        writer = csv.DictWriter(f, fieldnames=row.keys())
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)
                except Exception:
                    traceback.print_exc()

            try:
                calculate_auc_accuracy_to_txt(
                    input_csv_path=csv_file,
                    output_txt_path=os.path.join(training_args.output_dir, f"{training_args.output_csv_dir}_auc_accuracy_summary.txt")
                )
            except Exception:
                traceback.print_exc()
