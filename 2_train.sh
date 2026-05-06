# ---------------------------------------------------------------------------
# Balanced training data (Stage 2)
#
# Before submitting this job, generate the per-class balanced split:
#   python scripts/make_balanced_split.py \
#       --input-dir ./annotation \
#       --output    ./annotation/balanced_stage2_seed43.json \
#       --seed 43
# Stage 1 uses --seed 42; the different seed makes the two stages see
# different (but equally balanced) random samples.
#
# Replace the /path/to/* placeholders below with your local copies:
#   --model_name_or_path : MedGemma 4B-IT
#                          https://huggingface.co/google/medgemma-4b-it
#   --ecg_data_path      : MIMIC-IV-ECG WFDB root (PhysioNet credentialed)
#   --output_dir         : shared checkpoint directory for the full pipeline
#                          (must be the SAME path used by 1_train.sh; Stage 2
#                          reads Stage 1's checkpoint-*/full_model.bin here
#                          and writes its own LoRA checkpoints back to it)
# ---------------------------------------------------------------------------

deepspeed --include localhost:0,1 llava/train/2_train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --model_name_or_path /path/to/medgemma/ \
    --data_path ./annotation/balanced_stage2_seed43.json::./annotation/test.json \
    --ecg_data_path "/path/to/mimic-iv-ecg/files/" \
    --output_dir "/path/to/output/" \
    --ecg_encoder_dir "./llava/model/ecg_encoder/models/best.pt" \
    --tune_2_lora True \
    --mm_projector_lr 1e-5 \
    --deepspeed ./scripts/zero2.json \
    --ecg_tower d-beta \
    --seed 49 \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4
