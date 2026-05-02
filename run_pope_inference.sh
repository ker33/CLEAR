#!/bin/bash

echo "=================================================="
echo "🚀 开始进行 POPE 幻觉基准测试推理 (生成答案)..."
echo "=================================================="

# 确保输出答案的文件夹存在
mkdir -p ./playground/data/eval/pope/answers

# 运行推理程序
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-dualcd-lora-mix \
    --model-base ./checkpoints/llava-v1.5-7b-official \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-aaa.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "=================================================="
echo "✅ 推理完成！答案已保存在 answers 文件夹中。"
echo "=================================================="