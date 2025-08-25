#!/bin/bash
# run_inferencer.sh

set -e

# ─── Configuration ────────────────────────────────────────────────
ADV_PATH="/data/cs776/High_level_denoiser/cifar10_FGSM_adv.pt"
DATASET="cifar10"
MODEL_PATH="/data/cs776/High_level_denoiser/cifar10_classifier.pth"
SAVE_DIR="/data/cs776/High_level_denoiser/detector_save_${DATASET}"
ATTACK="FGSM"
MC_RUNS=50
THRESHOLD=0.32
NUM_SAMPLES=1000
BATCH_SIZE=128
SEED=42

# PHASE="test"
# feat_layer="pre_fc3"

LOG_DIR="log_infer_${DATASET}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/detector_infer_${DATASET}_${ATTACK}.log"

# ─── Run inference ───────────────────────────────────────────────
python main_integrated.py \
  --phase test \
  --attack      "${ATTACK}" \
  --feat_layer pre_fc3 \
  --adv_path    "${ADV_PATH}" \
  --dataset     "${DATASET}" \
  --model_path  "${MODEL_PATH}" \
  --save_dir    "${SAVE_DIR}" \
  --mc_runs     "${MC_RUNS}" \
  --threshold   "${THRESHOLD}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size  "${BATCH_SIZE}" \
  --seed        "${SEED}" | tee "${LOG_FILE}" \
