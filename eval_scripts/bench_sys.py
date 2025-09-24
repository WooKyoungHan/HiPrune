import os
os.environ['HIPRUNE_RETENTION'] = '192'
os.environ['HIPRUNE_ALPHA'] = '0.1'
os.environ['HIPRUNE_OBJECT_LAYER'] = '9'

import re
import torch
import time
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import load_images
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.model import *

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Describe this figure in detail."
image_file = "/root/HiPrune/LLaVA/images/llava_v1_5_radar.jpg"

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.measure_latency = True

qs = prompt
image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
if IMAGE_PLACEHOLDER in qs:
    if model.config.mm_use_im_start_end:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    else:
        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
else:
    if model.config.mm_use_im_start_end:
        qs = image_token_se + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

conv = conv_templates['llava_v1'].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image_files = image_file.split(",")
images = load_images(image_files)
image_sizes = [x.size for x in images]
images_tensor = process_images(
    images,
    image_processor,
    model.config
).to(model.device, dtype=torch.float16)

input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .cuda()
)

test_times = 10
prefill_time_list = []
decode_latency_list = []
peak_mem_list = []
throughput_list = []

with torch.inference_mode():
    for i in range(test_times + 1): # Exclude the first warmup
        model.reset_time()
        start = time.time()
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            max_new_tokens=500,
            use_cache=False,
        )[0]
        single_turn_time = time.time() - start
        prefill_time = model.prefill_latency
        decode_latency = model.decode_latency
        peak_memory_used = torch.cuda.max_memory_allocated("cuda:0") / (1024 * 1024 * 1024)
        throughput = output_ids.shape[-1] / single_turn_time
        
        prefill_time_list.append(prefill_time)
        decode_latency_list.append(decode_latency)
        peak_mem_list.append(peak_memory_used)
        throughput_list.append(throughput)

        print("="*30)
        print(f'prefill_time:{prefill_time:.2f}ms')
        print(f'decode_latence:{decode_latency:.2f}ms')
        print(f'peak_memory_used:{peak_memory_used:.2f}GB')
        print(f'throughput:{throughput}tokens/s')

print('----Final-----')
print(f'avg_prefill_time:{np.mean(prefill_time_list[1:]):.2f} +- {np.std(prefill_time_list[1:]):.2f}ms')
print(f'avg_decode_latence:{np.mean(decode_latency_list[1:]):.2f} +- {np.std(decode_latency_list[1:]):.2f}ms')
print(f'peak_memory_used:{np.mean(peak_mem_list[1:]):.2f} +- {np.std(peak_mem_list[1:]):.2f}GB')
print(f'avg_throughput:{np.mean(throughput_list[1:]) :.2f} +- {np.std(throughput_list[1:]):.2f}tokens/s')