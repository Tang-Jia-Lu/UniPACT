# ---------------------------------------------------------------------------
# Replace the /path/to/* placeholders below with your local copies:
#   --model_name_or_path : MedGemma 4B-IT
#                          https://huggingface.co/google/medgemma-4b-it
#   --ecg_data_path      : MIMIC-IV-ECG WFDB root (PhysioNet credentialed)
#   --output_dir         : shared checkpoint directory for the full pipeline
#                          (must be the SAME path used by 1_train.sh and
#                          2_train.sh; 3_test.sh reads the latest
#                          checkpoint-*/full_model.bin from here)
# ---------------------------------------------------------------------------

deepspeed --include localhost:0 llava/train/3_test.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --model_name_or_path /path/to/medgemma/ \
    --data_path ./annotation/test.json::./annotation/test.json \
    --ecg_data_path "/path/to/mimic-iv-ecg/files/" \
    --output_dir "/path/to/output/" \
    --output_csv_dir "deterioration_test_W_encoder_full_lora" \
    --ecg_encoder_dir "./llava/model/ecg_encoder/models/best.pt" \
    --mm_projector_lr 1e-5 \
    --deepspeed ./scripts/zero2.json \
    --ecg_tower d-beta \
    --seed 42 \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1 \
    --save_strategy "steps" \
    --save_steps 1 \
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
