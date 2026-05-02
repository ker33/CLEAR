#!/bin/bash

# 🌟 关键：完整的微调脚本，所有参数在一个命令中

deepspeed llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 256 \
    --lora_alpha 512 \
    --mm_projector_lr 5e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b-official \
    --version v1 \
    --data_path ./playground/data/llava_mixed_210k_ultimate.json \
    --image_folder ./playground/data/coco/train2014 \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-dualcd-lora-claude \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --max_grad_norm 0.5 \
    --report_to none \
    --seed 42
