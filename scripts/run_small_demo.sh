#!/bin/bash
# ============================================================
# Memory Dial: Small-Scale Reproducible Demo
# Reproduces qualitative behavior at α = {0.0, 0.5, 1.0}
# ============================================================

set -e

CONFIG=configs/default.yaml
MODEL=gpt2

ALPHAS=(0.0 0.5 1.0)

echo "============================================"
echo "Running Memory Dial small-scale demo"
echo "Model: $MODEL"
echo "Alphas: ${ALPHAS[@]}"
echo "============================================"

for ALPHA in "${ALPHAS[@]}"; do
  echo ""
  echo "--------------------------------------------"
  echo "Training model with alpha = $ALPHA"
  echo "--------------------------------------------"

  python experiments/train_memory_dial.py \
    --config $CONFIG \
    --model_name $MODEL \
    --alpha $ALPHA \
    --output_dir outputs/demo_alpha_${ALPHA}

  echo ""
  echo "Generating qualitative examples (alpha = $ALPHA)"

  python demo/generate_examples.py \
    --model_path outputs/demo_alpha_${ALPHA} \
    --alpha $ALPHA \
    --prompts demo/prompts.yaml \
    --output_file outputs/demo_alpha_${ALPHA}/samples.txt
done

echo ""
echo "============================================"
echo "Demo complete."
echo "Check outputs/demo_alpha_*/samples.txt"
echo "============================================"

