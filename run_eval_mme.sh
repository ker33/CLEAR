#!/bin/bash

EXP_NAME="llava-v1.5-13b-offi"

MODEL_PATH="./checkpoints/llava-v1.5-13b"
ROOT_DIR=$(pwd)
MME_DIR="$ROOT_DIR/playground/data/eval/MME"

echo "=========================================================="
echo "🚀 开始一键执行 MME 评测流程: $EXP_NAME"
echo "=========================================================="

echo "⏳ [1/3] 正在运行模型推理生成问答..."
mkdir -p "$MME_DIR/answers"

python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file "$MME_DIR/llava_mme.jsonl" \
    --image-folder "$MME_DIR/MME_Benchmark_release_version" \
    --answers-file "$MME_DIR/answers/$EXP_NAME.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "⏳ [2/3] 正在将 jsonl 转换为官方要求的 txt 格式..."
cd "$MME_DIR" || exit 1
python convert_answer_to_mme.py --experiment "$EXP_NAME"

echo "⏳ [3/3] 正在计算 Perception 和 Cognition 最终得分..."
# 进入打分工具目录
cd eval_tool || exit 1
python calculation.py --results_dir "answers/$EXP_NAME"

cd "$ROOT_DIR" || exit 1
echo "=========================================================="
echo "✅ $EXP_NAME MME 评测一键跑通！请查看上方输出的成绩。"
echo "=========================================================="
