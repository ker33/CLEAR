import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

# ==========================================================
# 🌟 创新注入 1：导入干预补丁与对比解码模块
# ==========================================================
from llava.model.attention_intervention import apply_mask_guided_intervention, MaskGuidedCDProcessor
from transformers.generation.logits_process import LogitsProcessorList


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # ==========================================================
    # 🌟 创新注入 2：挂载 PSI 注意力干预基础模块
    # ==========================================================
    print("🚀 正在激活 V8 掩码引导双径对比解码底层架构 (ScienceQA 兼容版)...")
    # dcd
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=15,   
    #     end_layer=28,     
    #     threshold=0.20,  
    #     alpha=1.0        
    # )
    # dcd-layer1
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=18,   
    #     end_layer=28,     
    #     threshold=0.20,  
    #     alpha=0.50        
    # )
    # dcd-layer2
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=20,   
    #     end_layer=28,     
    #     threshold=0.50,  
    #     alpha=0.80        
    # )
    # dcd-layer3
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=25,   
    #     end_layer=28,     
    #     threshold=0.20,  
    #     alpha=0.50        
    # )
    # dcd-layer4
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=18,   
    #     end_layer=28,     
    #     threshold=0.50,  
    #     alpha=1.0        
    # )
    # dcd-layer5
    apply_mask_guided_intervention(
        model=model, 
        start_layer=20,   
        end_layer=25,     
        threshold=0.50,  
        alpha=0.2        
    )
    # ==========================================================

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # ==========================================================
        # 🌟 创新注入 3：动态定位 <image> 位置并处理纯文本题目
        # ==========================================================
        logits_processor = None
        
        # 默认重置所有干预层的图片索引（防止纯文本题目被上一题干扰）
        for layer_idx in range(len(model.model.layers)):
            if hasattr(model.model.layers[layer_idx].self_attn, 'use_intervention'):
                model.model.layers[layer_idx].self_attn.img_start_idx = -1
                model.model.layers[layer_idx].self_attn.img_end_idx = -1
        
        # 如果当前题目有图片，则激活干预和对比解码
        if images is not None:
            image_token_pos = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
            if len(image_token_pos) > 0:
                img_start_idx = image_token_pos.item()
                img_end_idx = img_start_idx + 576 
                
                # 更新图片位置索引
                for layer_idx in range(len(model.model.layers)):
                    if hasattr(model.model.layers[layer_idx].self_attn, 'use_intervention'):
                        model.model.layers[layer_idx].self_attn.img_start_idx = img_start_idx
                        model.model.layers[layer_idx].self_attn.img_end_idx = img_end_idx
                
                # 实例化对比解码处理器
                cd_processor = MaskGuidedCDProcessor(
                    model=model,
                    images=images,
                    image_sizes=image_sizes,
                    penalty_alpha=0.4 # OWL 惩罚因子
                )
                logits_processor = LogitsProcessorList([cd_processor])
        # ==========================================================

        with torch.inference_mode():
            # 动态生成 kwargs，如果有图片则挂载 logits_processor
            gen_kwargs = {
                "inputs": input_ids,
                "images": images,
                "image_sizes": image_sizes,
                "do_sample": True if args.temperature > 0 else False,
                "temperature": args.temperature,
                "max_new_tokens": 1024,
                "use_cache": True,
            }
            
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor

            output_ids = model.generate(**gen_kwargs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)