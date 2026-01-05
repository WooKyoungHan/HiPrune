# #!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-13b"
# SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $1 \
#         --question-file /workspace/LLaVA/playground/data/eval/vqav2/$SPLIT.jsonl \
#         --image-folder /workspace/LLaVA/playground/data/eval/vqav2/test2015 \
#         --answers-file /workspace/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/workspace/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /workspace/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

#!/bin/bash
set -euo pipefail
set -x

set -e

cd /workspace/LLaVA   # ★ 중요: convert 스크립트 상대경로 기준 맞추기

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

# answers 폴더 미리 생성 (chunk 파일 쓰기 전에)
# LLaVA/playground/data/eval/vqav2
mkdir -p /workspace/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$1" \
        --question-file /workspace/LLaVA/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /workspace/LLaVA/playground/data/eval/vqav2/test2015 \
        --answers-file /workspace/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/workspace/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /workspace/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# convert가 ./playground/...를 쓰는 경우가 많아서, 여기서도 repo root에서 실행되는 게 중요
python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT
