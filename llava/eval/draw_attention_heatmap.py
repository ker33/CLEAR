import argparse
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# ❗先默认关闭干预（调试阶段必须关）
# from llava.model.attention_intervention import apply_mask_guided_intervention


# ===============================
# 🔥 可视化函数（增强版）
# ===============================
def draw_and_save_heatmap(image_path, attention_weights, save_path):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    h, w, _ = original_image.shape

    print("\n📊 Attention Stats:")
    print("min:", attention_weights.min())
    print("max:", attention_weights.max())
    print("mean:", attention_weights.mean())

    # reshape
    attn_grid = attention_weights.reshape(24, 24)

    # 🚨 防塌缩
    if attn_grid.max() - attn_grid.min() < 1e-6:
        print("⚠️ Attention几乎无变化（塌缩），强行拉伸")
        attn_grid = np.ones_like(attn_grid) * 0.5
    else:
        attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min())

    # 🔥 gamma增强（关键！）
    attn_grid = np.power(attn_grid, 0.4)

    # resize
    attn_grid = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_CUBIC)

    # 平滑
    attn_grid = cv2.GaussianBlur(attn_grid, (51, 51), 0)

    # 再归一化
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-9)
    attn_grid = np.uint8(255 * attn_grid)

    heatmap = cv2.applyColorMap(attn_grid, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    print(f"✅ 热力图已保存至: {save_path}")


# ===============================
# 🔥 主函数
# ===============================
def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # ❗调试阶段先不要开干预
    # print("🚀 激活干预机制...")
    # apply_mask_guided_intervention(model=model, start_layer=8, end_layer=28, threshold=0.50, alpha=0.5)

    image = Image.open(args.image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()

    qs = args.question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    # ===============================
    # 🔥 找 image token 范围
    # ===============================
    image_token_pos = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
    img_start_idx = image_token_pos.item()
    img_end_idx = img_start_idx + 576

    print(f"📍 Image token range: {img_start_idx} - {img_end_idx}")

    # ===============================
    # 🔥 推理 + attention
    # ===============================
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            max_new_tokens=10,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

    generated_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
    print(f"\n🤖 模型回答: {generated_text}")

    # ===============================
    # 🔥 核心改动：用 Prefill 阶段
    # ===============================
    target_attn = outputs.attentions[0]   # ✅ 关键！！
    q_idx = -1                            # 最后一个token

    print("🎯 使用 Prefill 阶段 attention")

    # ===============================
    # 🔥 选中层（更合理）
    # ===============================
    selected_layers_attn = []

    for layer_idx in range(5, 15):   # ✅ 改这里！！
        layer_attn = target_attn[layer_idx]

        mean_head_attn = layer_attn[0, :, q_idx, img_start_idx:img_end_idx] \
            .mean(dim=0) \
            .to(torch.float32) \
            .cpu()

        print(f"Layer {layer_idx}: min={mean_head_attn.min():.6f}, max={mean_head_attn.max():.6f}")

        selected_layers_attn.append(mean_head_attn)

    final_visual_attn = torch.stack(selected_layers_attn).mean(dim=0).numpy()

    # ===============================
    # 🔥 画图
    # ===============================
    draw_and_save_heatmap(
        args.image_path,
        final_visual_attn,
        args.output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="heatmap_output.jpg")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")

    args = parser.parse_args()
    main(args)