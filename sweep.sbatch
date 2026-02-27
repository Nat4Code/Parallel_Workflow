#!/usr/bin/env bash
#SBATCH --job-name=tile_opt
#SBATCH -A ISAAC-UTK0448
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --job-name=jam_array
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-15
#SBATCH --time=00:02:00
#SBATCH --mem=500M

set -euo pipefail

mkdir -p logs results

NX=4
NY=4

# the real array index, e.g. (0..15)
ARRAY_ID="${SLURM_ARRAY_TASK_ID}"

# task-id we pass to python (may be overridden for failure)
TID="${ARRAY_ID}"

HOST="$(hostname)"
TS_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Per-task parameterization (different input/parameter/seed):
SEED=$((1000 + ARRAY_ID))            # distinct seed per task

# Informatics requested:
echo "RUN_HEADER task_id=${ARRAY_ID} seed=${SEED} timestamp_utc=${TS_UTC} hostname=${HOST}"

# Intentional Failure: Out-of-bounds task_id.
if [ "$ARRAY_ID" -eq 12 ]; then
    TID=100  # out-of-bounds index for 4x4 grid (max index 15)
    echo "Causing intentional failure:"
    echo "We set TID to $TID which is out of bounds for our 4x4 grid (max index 15)."
    echo "Demonstrates workflow error handling using an explicit sanity-check trigger."
    echo "Our Python handles this with a ValueError and exits before wasting compute units."
    echo
    echo "To fix this, either remove the intentional failure block in run_worker.py"
    echo "...or skip this task in the sweep.sbatch array definition"
    echo " (e.g., sbatch --array=0-11,13-15 sweep.sbatch)."
    echo
    echo "To rerun the task, use: sbatch --array=12 sweep.sbatch"
    echo 
fi

# run inside container:
SIF="worker_py.sif"
if [ ! -f "$SIF" ]; then
  echo "ERROR: missing container image: $SIF"
  echo "Fix: apptainer build worker_py.sif worker_py.def"
  exit 3
fi

# run script in parallel...
apptainer exec --cleanenv \
  --bind "$PWD:/work" \
  "$SIF" \
  bash -lc "cd /work && python3 run_worker.py \
    --task-id \"$TID\" \
    --nx \"$NX\" --ny \"$NY\" \
    --steps 60000 \
    --coarse-gx 25 --coarse-gy 25 \
    --dt 0.02 --mass 1.0 --gamma 0.8 \
    --seed \"$SEED\" \
    --outdir results"

echo "Completed task $TID"