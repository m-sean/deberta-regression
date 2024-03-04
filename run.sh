DATA=~/dev/datasets/intensity/
MODEL=microsoft/deberta-base
MODEL_SAVE_DIR=deberta-intensity
EPOCHS=2

python3.8 main.py \
    --train \
    --lr=1e-5 \
    --loss=huber\
    --model-dir=$MODEL_SAVE_DIR \
    --data-dir=$DATA \
    --model-name=$MODEL \
    --epochs=$EPOCHS \
    --batch-size=8 \