#!/bin/bash
# ================= 配置区 =================
EXP_NAME="llava-v1.5-7b-dualcd-lora-dcd-layer3"
MODEL_PATH="./checkpoints/llava-v1.5-7b-dualcd-lora-mix"
BASE_PATH="./checkpoints/llava-v1.5-7b-official"
# ==========================================

VQA_DIR="./playground/data/eval/vqav2"
mkdir -p "$VQA_DIR/answers"

# echo "=========================================================="
# echo "🚀 开始执行 VQAv2 评测及打包流程"
# echo "=========================================================="

# echo "⏳ [1/2] 正在运行 VQAv2 模型推理 (⚠️警告: 约 10.7万题，请耐心等待!)..."
# python -m llava.eval.model_vqa_loader \
#     --model-path "$MODEL_PATH" \
#     --model-base "$BASE_PATH" \
#     --question-file "$VQA_DIR/llava_vqav2_mscoco_test-dev2015.jsonl" \
#     --image-folder "$VQA_DIR/test2015" \
#     --answers-file "$VQA_DIR/answers/$EXP_NAME.jsonl" \
#     --temperature 0 \
#     --conv-mode vicuna_v1

echo "⏳ [2/2] 正在转换为 EvalAI 官网提交格式..."
python scripts/convert_vqav2_for_submission.py \
    --split llava_vqav2_mscoco_test-dev2015 \
    --ckpt "$EXP_NAME" \
    --dir "$VQA_DIR"

echo "=========================================================="
echo "✅ 打包完毕！"
echo "👉 你的提交文件已保存在："
echo "   $VQA_DIR/llava_vqav2_mscoco_test-dev2015/$EXP_NAME.json"
echo "👉 请将此文件下载到本地，前往 EvalAI (VQA Challenge 2017) 提交获取 Test-dev 分数！"
echo "=========================================================="