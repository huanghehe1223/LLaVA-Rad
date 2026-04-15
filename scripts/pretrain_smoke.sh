#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

model_base=/cxr-report/model/vicuna-7b-v1.5
# output_dir="${1:-./checkpoints}"
output_dir="${1:-/cxr-report/smoke_output}"

data_path=/cxr-report/json_files/test.json

loader="mimic_train_findings"

image_folder=/cxr-report/dataset


################## Run name ##################
vision_tower="rad-dino"
vision_tower_config="/cxr-report/model/rad-dino/dinov2"
vision_tower_checkpoint="/cxr-report/model/rad-dino/backbone_compatible.safetensors"

epoch="${2:-1}"
bsz="${3:-4}"
grad_acc="${4:-2}"
lr="1e-3"
schedule="pt-${epoch}e"
run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
################## Run name ##################


# Global batch size should be 256

WANDB_RUN_ID="llava-pt-$(date +%Y%m%d%H%M%S)" WANDB_PROJECT="llava" WANDB_RUN_GROUP=pre-train \
    deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${model_base} \
    --version plain \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${grad_acc} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 3 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${run_name}
