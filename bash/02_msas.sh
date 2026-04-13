#!/bin/bash
#SBATCH --job-name=foldfit-msas
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/msas_%j.out
#SBATCH --error=logs/msas_%j.err

# ============================================================
# Step 2: Compute MSAs for antibody sequences
# ============================================================
# CPU-only. No GPU needed. High memory recommended for large DBs.
#
# Methods (set in config.yaml under msa.method):
#   - mmseqs:    Fast (~5s/query). Requires MMseqs2 + local DB.
#   - jackhmmer: Slow (~min/query). Requires HMMER3 + FASTA DBs.
#   - colabfold: Medium (~30s/query). Free API, no local DBs needed.
#
# Estimated time:
#   - mmseqs:    ~30 min for 500 structures
#   - jackhmmer: ~8 hours for 500 structures
#   - colabfold: ~4 hours for 500 structures (rate-limited)
#
# Prerequisites:
#   - Step 1 (preprocessing) completed
#   - Database paths configured in config.yaml
# ============================================================

set -euo pipefail

# -- Edit these paths --
REPO_DIR="<PATH_TO_FINETUNING_DIR>"
OPENFOLD_DIR="<PATH_TO_OPENFOLD3_DIR>"
# ----------------------

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONPATH="${REPO_DIR}/..:${OPENFOLD_DIR}"

python -m finetuning.scripts.compute_msas \
    --config config.yaml

echo "MSA computation complete."
echo "Next step: sbatch bash/03_train.sh"
