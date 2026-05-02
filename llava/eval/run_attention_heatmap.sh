python ./llava/eval/draw_attention_heatmap.py \
    --model-path ./checkpoints/llava-v1.5-7b-official \
    --image-path ./playground/data/coco/val2014/COCO_val2014_000000000923.jpg \
    --question "Is there a train in the image?" \
    --output-path "heatmap_train.jpg"