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
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

# ==========================================================
# 🌟 创新注入 1：导入我们的注意力干预补丁
# ==========================================================
from llava.model.attention_intervention import apply_mask_guided_intervention


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    #att-1
    # ==========================================================
    # 🌟 创新注入 2：挂载隐空间因果解耦注意力干预模块
    # ==========================================================
    print("🚀 正在激活隐空间因果解耦注意力干预 (DualCD-Intervention)...")
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=8, # 经验值：干预 15~32 层 (中高层负责具象语义，低层保留原始特征)
    #     end_layer=30, 
    #     threshold=1.0,  # 诊断阈值：当 看图得分 < 1.0 * 看字得分 时启动干预
    #     alpha=0.4,      # 靶向干预：物体区域的注意力增强系数
    #     b=0.15           # 靶向干预：历史文本的注意力抑制系数
    # )
    # att-2
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=15,  # 💡 退回15层：让1~14层先完成基础视觉提取和 OCR 识别
    #     end_layer=30,    # 避开最后的映射层
    #     threshold=0.15,  # 💡 黄金阈值：当模型分配给图片的注意力低于 15% 时，才触发保护
    #     alpha=0.3        # 💡 取消了 b 参数。纯粹通过靶向增强解救幻觉。
    # )
    # att-3
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=15, 
    #     end_layer=31,    # 放宽到 31 层，让深层语义也能受到无损修正
    #     threshold=0.12,  # 💡 将触发阈值微微下调至 12%，给 LLM 留出充足的思考空间
    #     alpha=0.4        
    # )
    # att-4
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=15, 
    #     end_layer=30,    
    #     threshold=0.15,  
    #     alpha=0.4        # 使用 0.4 的标准力度配合 0.5 的柔和掩码，达到完美平衡
    # )
    # att-5
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=15, 
    #     end_layer=30,    
    #     threshold=0.20,  # 只要图片注意力低于 20% 就触发唤醒
    #     alpha=1.5        # 💡 使用对数平移，+1.5 相当于概率放大 4.5倍，无损且强劲
    # )
    #att-6-1 #att-6 threshold=0.20
    # apply_mask_guided_intervention(
    #     model=model, 
    #     start_layer=15, 
    #     end_layer=30,    
    #     threshold=0.50,  
    #     alpha=1.0        # 💡 在概率域操作，1.0 足以产生脱胎换骨的变化
    # )
    # att-6-layer
    # apply_mask_guided_intervention(
    # model=model, 
    # start_layer=12,   # 避开底层，保护视觉基础特征
    # end_layer=28,     # 避开顶层，保护语言输出流畅度
    # threshold=0.50,  
    # alpha=1.0        
    # )
    # att-6-layer-1
    # apply_mask_guided_intervention(
    # model=model, 
    # start_layer=16,   # 避开底层，保护视觉基础特征
    # end_layer=28,     # 避开顶层，保护语言输出流畅度
    # threshold=0.50,  
    # alpha=1.0        
    # )
    # att-6-layer-2
    apply_mask_guided_intervention(
    model=model, 
    start_layer=18,   # 避开底层，保护视觉基础特征
    end_layer=28,     # 避开顶层，保护语言输出流畅度
    threshold=0.50,  
    alpha=1.0        
    )
    # ==========================================================

    questions =[json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        # ==========================================================
        # 🌟 创新注入 3：动态定位当前数据的 <image> 位置 (精简版)
        # ==========================================================
        image_token_pos = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
        
        if len(image_token_pos) > 0:
            img_start_idx = image_token_pos.item()
            # LLaVA-v1.5 的图片被编码为 576 个 patch
            img_end_idx = img_start_idx + 576 
            
            # 批量更新所有干预层的图片索引，文本区间由底层自动切分！
            for i in range(len(model.model.layers)):
                if hasattr(model.model.layers[i].self_attn, 'use_intervention'):
                    attn = model.model.layers[i].self_attn
                    attn.img_start_idx = img_start_idx
                    attn.img_end_idx = img_end_idx
        # ==========================================================

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)