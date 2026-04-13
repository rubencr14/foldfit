#!/bin/bash
# ============================================================
# Submit the full FoldFit pipeline as dependent SLURM jobs
# ============================================================
# Usage: bash bash/run_all.sh
#
# Submits 3 jobs with dependencies:
#   preprocess -> msas -> train
# Each job starts only after the previous one succeeds.
# ============================================================

set -euo pipefail

echo "Submitting FoldFit pipeline..."

# Step 1: Preprocess
JOB1=$(sbatch --parsable bash/01_preprocess.sh)
echo "Step 1 (preprocess):  Job ${JOB1}"

# Step 2: MSAs (depends on step 1)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} bash/02_msas.sh)
echo "Step 2 (MSAs):        Job ${JOB2} (after ${JOB1})"

# Step 3: Train (depends on step 2)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} bash/03_train.sh)
echo "Step 3 (train):       Job ${JOB3} (after ${JOB2})"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Logs will be in: logs/"
