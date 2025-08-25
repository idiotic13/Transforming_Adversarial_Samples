#!/bin/bash

# Make sure script exits if any command fails
set -e

# Define arguments
DATASET="mnist"                     # or "cifar"                     # or "pgd", "bim-a", etc.
BATCH_SIZE=256
MODEL_PATH="/data/cs776/detector/saved_models/${DATASET}_classifier.pth" # path to teh classifier pretrained models

# Run the script
for ATTACK in FGSM PGD bim-a bim-b; do
    DATA_PATH="/data/cs776/detector/adv_data/${DATASET}_${ATTACK}_adv.pt"
    SAVE_DIR="/data/cs776/detector/detector_save_${DATASET}_${ATTACK}"
    LOG_DIR="log_${DATASET}"

    mkdir -p "$SAVE_DIR"
    mkdir -p "$LOG_DIR"

    LOG_FILE="${LOG_DIR}/detector_train_${DATASET}_${ATTACK}.log"

    echo "=== Running ${DATASET}/${ATTACK} ==="
    
    python detector_training.py \
    -d $DATASET \
    -a $ATTACK \
    -b $BATCH_SIZE \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR | tee "$LOG_FILE"
done