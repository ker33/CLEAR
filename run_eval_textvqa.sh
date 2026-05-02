#!/bin/bash
# ================= 配置区 =================
EXP_NAME="llava-v1.5-7b-dualcd-lora-dcd-layer5"
MODEL_PATH="./checkpoints/llava-v1.5-7b-dualcd-lora-mix"
BASE_PATH="./checkpoints/llava-v1.5-7b-official"
# ==========================================

TVQA_DIR="./playground/data/eval/textvqa"
mkdir -p "$TVQA_DIR/answers"

echo "=========================================================="
echo "🚀 开始执行 TextVQA 评测流程"
echo "=========================================================="

echo "⏳ [1/2] 正在运行 TextVQA 模型推理..."
# TextVQA 用通用的加载器即可
python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --model-base "$BASE_PATH" \
    --question-file "$TVQA_DIR/llava_textvqa_val_v051_ocr.jsonl" \
    --image-folder "$TVQA_DIR/images/train_images" \
    --answers-file "$TVQA_DIR/answers/$EXP_NAME.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "⏳[2/2] 正在计算 TextVQA 得分..."
# 调用 LLaVA 提供的 TextVQA 算分脚本
python -m llava.eval.eval_textvqa \
    --annotation-file "$TVQA_DIR/TextVQA_0.5.1_val.json" \
    --result-file "$TVQA_DIR/answers/$EXP_NAME.jsonl"

echo "=========================================================="
echo "✅ TextVQA 评测完成！请看终端上方打印的 Accuracy 得分！"
echo "=========================================================="