#!/bin/bash

echo "=================================================="
echo "📊 开始对生成的答案进行 POPE 幻觉指标打分..."
echo "=================================================="

# 运行官方的评分程序
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-aaa.jsonl

echo "=================================================="
echo "🎉 ttt打分结束！请记录终端输出的 Accuracy, F1-Score 和 Yes-Ratio！"
echo "=================================================="