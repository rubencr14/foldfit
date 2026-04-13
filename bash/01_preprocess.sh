#!/bin/bash
#SBATCH --job-name=foldfit-preprocess
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# ============================================================
# Step 1: Fetch and preprocess antibody structures from RCSB PDB
# ============================================================
# This is CPU-only. No GPU needed.
# Estimated time: ~5 min for 100 structures (single-threaded)
#
# What it does:
#   - Downloads mmCIF files from RCSB PDB
#   - Preprocesses structures: tokenization, bond detection, conformers
#   - Creates dataset cache (ClusteredDatasetCache format for OF3)
#
# Prerequisites:
#   - CCD downloaded to ~/.openfold3/chemical_component_dictionary.cif
#   - OpenFold3 repo cloned at ../openfold-3
# ============================================================

set -euo pipefail

# -- Edit these paths --
REPO_DIR="<PATH_TO_FINETUNING_DIR>"       # e.g. /home/user/foldfit/finetuning
OPENFOLD_DIR="<PATH_TO_OPENFOLD3_DIR>"    # e.g. /home/user/openfold-3
CCD_PATH="<PATH_TO_CCD>"                 # e.g. ~/.openfold3/chemical_component_dictionary.cif
# ----------------------

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONPATH="${REPO_DIR}/..:${OPENFOLD_DIR}"

python -m finetuning.scripts.prepare_antibody_data \
    --config config.yaml \
    --output-dir ./data/antibody_training \
    --ccd-path "${CCD_PATH}" \
    --max-structures 500 \
    --max-resolution 3.0 \
    --num-workers 0

echo "Preprocessing complete."
echo "Next step: sbatch bash/02_msas.sh"
