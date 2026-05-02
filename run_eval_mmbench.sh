#!/bin/bash
# ================= 配置区 =================
EXP_NAME="llava-v1.5-7b-dualcd-lora-dcd-layer1"
MODEL_PATH="./checkpoints/llava-v1.5-7b-dualcd-lora-mix"
# 如果你的模型不需要 base，请把下面这行删掉，并把第 20 行的 --model-base 删掉
BASE_PATH="./checkpoints/llava-v1.5-7b-official"
# ==========================================

MMB_DIR="./playground/data/eval/mmbench"

# echo "=========================================================="
# echo "🚀 开始执行 MMBench 评测流程"
# echo "=========================================================="

# echo "⏳ [1/2] 正在运行 MMBench 模型推理 (这步大概需要十几分钟)..."
# mkdir -p "$MMB_DIR/answers"

# python -m llava.eval.model_vqa_mmbench \
#     --model-path "$MODEL_PATH" \
#     --model-base "$BASE_PATH" \
#     --question-file "$MMB_DIR/mmbench_dev_20230712.tsv" \
#     --answers-file "$MMB_DIR/answers/$EXP_NAME.jsonl" \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1


echo "⏳ [2/2] 正在将 jsonl 结果打包成 MMBench 官网所需的 Excel 提交格式..."
mkdir -p "$MMB_DIR/answers_upload"

# 运行 LLaVA 提供的打包脚本
python scripts/convert_mmbench_for_submission.py \
    --annotation-file "$MMB_DIR/mmbench_dev_20230712.tsv" \
    --result-dir "$MMB_DIR/answers" \
    --upload-dir "$MMB_DIR/answers_upload" \
    --experiment "$EXP_NAME"

echo "=========================================================="
echo "✅ 打包完毕！"
echo "👉 请将服务器上的这个文件下载到你的本地电脑："
echo "   $MMB_DIR/answers_upload/$EXP_NAME.xlsx"
echo "=========================================================="