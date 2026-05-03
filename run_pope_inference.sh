#!/bin/bash

mkdir -p ./playground/data/eval/pope/answers

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-dualcd-lora-mix \
    --model-base ./checkpoints/llava-v1.5-7b-official \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
