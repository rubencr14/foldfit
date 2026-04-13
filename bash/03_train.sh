#!/bin/bash
#SBATCH --job-name=foldfit-train
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================
# Step 3: LoRA fine-tuning of OpenFold3
# ============================================================
# Requires GPU. Memory depends on token_budget:
#
#   token_budget=48  -> ~14 GB VRAM (RTX 3090/4090)
#   token_budget=128 -> ~28 GB VRAM (A100 40GB)
#   token_budget=256 -> ~50 GB VRAM (A100 80GB)
#   token_budget=384 -> ~70 GB VRAM (H100 80GB)
#
# Set token_budget in config.yaml or override via CLI.
#
# Prerequisites:
#   - Step 1 (preprocessing) completed
#   - Step 2 (MSAs) completed
#   - Pretrained weights downloaded
# ============================================================

set -euo pipefail

# -- Edit these paths --
REPO_DIR="<PATH_TO_FINETUNING_DIR>"
OPENFOLD_DIR="<PATH_TO_OPENFOLD3_DIR>"
# ----------------------

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONPATH="${REPO_DIR}/..:${OPENFOLD_DIR}"

# Optional: load modules on your cluster
# module load cuda/12.1

python -m finetuning.scripts.train_lora \
    --config config.yaml

echo "Training complete."
echo "LoRA weights saved to: $(grep output_dir config.yaml | awk '{print $2}')/lora_final.pt"
