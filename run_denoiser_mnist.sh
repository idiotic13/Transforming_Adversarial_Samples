#!/bin/bash
# ================================
# Script to train and test the denoiser
# on a given dataset and feature layer
# for multiple adversarial attacks
# ================================

# ------------------------
# Set common parameters
# ------------------------
# DATASET: which dataset to use 
DATASET="mnist"
# FEAT_LAYER: which classifier layer to compute feature loss on
FEAT_LAYER="pre_fc2"

# ------------------------
# Define list of attacks
# ------------------------
# ATTACKS: array of adversarial methods to iterate over
ATTACKS=("FGSM" "PGD" "bim-a" "bim-b")

# ------------------------
# Loop over each attack
# ------------------------
for ATTACK in "${ATTACKS[@]}"; do
    # Print status message for training
    echo "==== Running TRAIN for attack: $ATTACK ===="
    # Invoke training phase
    python3 main.py \
        --phase train \
        --attack "$ATTACK" \
        --dataset "$DATASET" \
        --feat_layer "$FEAT_LAYER" \
        --epochs 30

    # Print status message for testing
    echo "==== Running TEST for attack: $ATTACK ===="
    # Invoke testing phase
    python3 main.py \
        --phase test \
        --attack "$ATTACK" \
        --dataset "$DATASET" \
        --feat_layer "$FEAT_LAYER" \
        --epochs 60
done