#!/bin/bash

set -e
set -o pipefail

model_base=/cxr-report/model/vicuna-7b-v1.5
model_path=microsoft/llava-rad

model_base="${1:-$model_base}"
model_path="${2:-$model_path}"
prediction_dir="${3:-/cxr-report/llavarad}"
prediction_file=$prediction_dir/test
merged_prediction_file=$prediction_dir/mimic_cxr_preds.jsonl

run_name="${4:-llavarad}"


query_file=/cxr-report/json_files/test.json

image_folder=/cxr-report/dataset
loader="mimic_test_findings"
conv_mode="v1"

CHUNKS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

mkdir -p "${prediction_dir}"

for (( idx=0; idx<$CHUNKS; idx++ ))
do
    CUDA_VISIBLE_DEVICES=$idx python -m llava.eval.model_mimic_cxr \
        --query_file ${query_file} \
        --loader ${loader} \
        --image_folder ${image_folder} \
        --conv_mode ${conv_mode} \
        --prediction_file ${prediction_file}_${idx}.jsonl \
        --temperature 0 \
        --model_path ${model_path} \
        --model_base ${model_base} \
        --chunk_idx ${idx} \
        --num_chunks ${CHUNKS} \
        --batch_size 8 \
        --group_by_length &
done

wait

cat ${prediction_file}_*.jsonl > "${merged_prediction_file}"

pushd llava/eval/rrg_eval
WANDB_PROJECT="llava" WANDB_RUN_ID="llava-eval-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=evaluate CUDA_VISIBLE_DEVICES=0 \
    python run.py "${merged_prediction_file}" --run_name ${run_name} --output_dir "${prediction_dir}/eval"
popd
