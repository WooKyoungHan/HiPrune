# conda activate HiPrune
cd lmms-eval
pip install av==14.4.0
pip install -U -e .
pip install openpyxl==3.1.5 qwen_vl_utils

# IMPORTANT: Since llava needs an old version of transformers, which is not compatible with Qwen
# Before evaluating Qwen, you should first upgrade transformers and accelerate

pip3 install qwen_vl_utils
pip install transformers==4.52.0 accelerate
pip install flash_attn # FlashAttention is required