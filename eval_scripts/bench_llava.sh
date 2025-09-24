export HF_ENDPOINT=https://hf-mirror.com # If you encounter network issue, please uncomment this
export CUDA_VISIBLE_DEVICES=0

export HIPRUNE_RETENTION=192 # Number of tokens to be retained
export HIPRUNE_ALPHA=0.1
export HIPRUNE_OBJECT_LAYER=9

ckpt=liuhaotian/llava-v1.5-7b

#The following benchmarks are obtained with lmm_eval: gqa, mme, pope, scienceqa_img, vizwiz_vqa_val
tasks=("scienceqa_img" "vizwiz_vqa_val")
for task in "${tasks[@]}"; do
    accelerate launch --num_processes=1 -m lmms_eval --model llava \
        --model_args pretrained=$ckpt \
        --tasks "$task" --batch_size 1 --log_samples \
        --log_samples_suffix reproduce --output_path ./logs/
done

cd LLaVA
# For MMB, MMBCN, VQAv2, and TextVQA, we use the scripts provided in LLaVA
# Please prepare data following the evaluation instrution first

# Metrics that need online evaluation
bash scripts/v1_5/eval/mmbench.sh $ckpt
bash scripts/v1_5/eval/mmbench_cn.sh $ckpt
# Metrics that are extremely slow or in correct with LMMs-Eval
bash scripts/v1_5/eval/textvqa.sh $ckpt
bash scripts/v1_5/eval/vqav2.sh $ckpt
