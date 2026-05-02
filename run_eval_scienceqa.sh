#!/bin/bash
EXP_NAME="llava-v1.5-13b-offi"
MODEL_PATH="./checkpoints/llava-v1.5-13b"
SQA_DIR="./playground/data/eval/scienceqa"

echo "⏳ [1/2] 开始 ScienceQA 推理..."
# 注意：ScienceQA 有专属的 Loader
python -m llava.eval.model_vqa_science \
    --model-path "$MODEL_PATH" \
    --model-base "$BASE_PATH" \
    --question-file "$SQA_DIR/llava_test_CQM-A.json" \
    --image-folder "$SQA_DIR/images/test" \
    --answers-file "$SQA_DIR/answers/$EXP_NAME.jsonl" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "⏳ [2/2] 开始计算 ScienceQA 得分..."
python -m llava.eval.eval_science_qa \
    --base-dir "$SQA_DIR" \
    --result-file "$SQA_DIR/answers/$EXP_NAME.jsonl" \
    --output-file "$SQA_DIR/answers/${EXP_NAME}_output.jsonl" \
    --output-result "$SQA_DIR/answers/${EXP_NAME}_result.json"

echo "✅ 评测完成！详细成绩已保存在 answers 文件夹中。"