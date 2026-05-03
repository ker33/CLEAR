#!/bin/bash

EXP_NAME="llava-v1.5-7b"

MODEL_PATH="./checkpoints/llava-v1.5-7b-dualcd-lora-mix"
BASE_PATH="./checkpoints/llava-v1.5-7b-official" 

ROOT_DIR=$(pwd)
CHAIR_DIR="$ROOT_DIR/playground/data/eval/CHAIR"
COCO_IMG_DIR="$ROOT_DIR/playground/data/coco/val2014"
TOOL_DIR="$CHAIR_DIR/chair_tool"

mkdir -p "$CHAIR_DIR/answers"

echo "⏳ [0/4] 正在准备 NLTK 环境和测试提问文件..."
pip install nltk >/dev/null 2>&1
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

if [ ! -f "$CHAIR_DIR/chair_questions.jsonl" ]; then
python - <<EOF
import os, json, random
imgs = sorted([f for f in os.listdir("$COCO_IMG_DIR") if f.endswith('.jpg')])
random.seed(42) 
sampled = random.sample(imgs, 500)
with open("$CHAIR_DIR/chair_questions.jsonl", "w") as f:
    for img in sampled:
        img_id = int(img.split('_')[-1].split('.')[0])
        f.write(json.dumps({"question_id": img_id, "image": img, "text": "Please describe this image in detail."}) + '\n')
EOF
echo "   ✅ 测试提问文件生成完毕 (500张图)！"
fi

echo "⏳ [1/4] 正在运行模型推理 (生成图片描述)..."
python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --model-base "$BASE_PATH" \
    --question-file "$CHAIR_DIR/chair_questions.jsonl" \
    --image-folder "$COCO_IMG_DIR" \
    --answers-file "$CHAIR_DIR/answers/$EXP_NAME.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "⏳ [2/4] 正在将输出结果转换为 CHAIR 格式..."
python - <<EOF
import json
out =[]
with open("$CHAIR_DIR/answers/$EXP_NAME.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        out.append({"image_id": d["question_id"], "caption": d["text"]})
with open("$TOOL_DIR/${EXP_NAME}_caps.json", "w") as f:
    json.dump(out, f, indent=4)
EOF

echo "⏳ [3/4] 正在计算最终 CHAIR 幻觉指标..."
cd "$TOOL_DIR" || exit 1
python chair.py \
    --cap_file "${EXP_NAME}_caps.json" \
    --coco_path "annotations"

cd "$ROOT_DIR" || exit 1
echo "=========================================================="
echo "✅ $EXP_NAME CHAIR 评测完成！请查看上方的 CHAIR_i 和 CHAIR_s 得分（越低越好）。"
echo "=========================================================="
