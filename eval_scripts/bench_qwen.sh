# IMPORTANT: Since llava needs an old version of transformers, which is not compatible with Qwen
# Before evaluating Qwen, you should first upgrade transformers and accelerate

# pip3 install qwen_vl_utils
# pip install transformers==4.52.0 accelerate
# pip install flash_attn # FlashAttention is required

export HF_ENDPOINT=https://hf-mirror.com # If you encounter network issue, please uncomment this
export CUDA_VISIBLE_DEVICES=0

export HIPRUNE_QWEN_RETENTION=0.223
export HIPRUNE_ALPHA=0.1
export HIPRUNE_OBJECT_LAYER=16

tasks=("mmbench_en_dev" "mmbench_cn_dev" "pope" "scienceqa_img" "vizwiz_vqa_val")
for task in "${tasks[@]}"; do
    accelerate launch --num_processes=1 -m lmms_eval --model qwen2_5_vl \
        --model_args pretrained="Qwen/Qwen2.5-VL-3B-Instruct",attn_implementation="flash_attention_2" \
        --tasks "$task" --batch_size 1 --log_samples \
        --log_samples_suffix reproduce --output_path ./logs/
done