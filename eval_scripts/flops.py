'''
If you encounter ValueError: too many values to unpack (expected 6)
You can revise line 97-104 in LLaVA/llava/model/language_model/llava_llama.py to
    (
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
        _, # Add this line
        _ # Add this line
    ) = self.prepare_inputs_labels_for_multimodal(
This is because the original code in LLaVA does not go in this line and actually have some bugs
Nevertheless, we keep the original code in the repo for consistency with LLaVA
'''
import os
os.environ['HIPRUNE_RETENTION'] = '192'
os.environ['HIPRUNE_ALPHA'] = '0.1'
os.environ['HIPRUNE_OBJECT_LAYER'] = '9'

import torch
import re
from calflops import calculate_flops
from transformers import AutoModel
from transformers import AutoTokenizer

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
image_file = "LLaVA/images/llava_v1_5_radar.jpg"

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name,
    torch_dtype=torch.bfloat16
)

batch_size = 1
max_seq_length = 128

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
    
inputs = {'input_ids': input_ids,
          'images': images_tensor,
          'image_sizes': image_sizes}

flops, macs, params = calculate_flops(model=model,
                                      kwargs = inputs,
                                      print_results=False)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))