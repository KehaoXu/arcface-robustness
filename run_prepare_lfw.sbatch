#!/bin/bash
#SBATCH --job-name=prepare-lfw
#SBATCH --partition=a100
#SBATCH --qos=normal
#SBATCH --account=peirong
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module load anaconda3
conda activate face

echo "========== Job Info =========="
echo "Host: $(hostname)"
echo "Project root: ${SLURM_SUBMIT_DIR}"
echo "Command: python prepare_lfw.py --input-dir lfw/lfw-deepfunneled/lfw-deepfunneled --output-dir dataset/original"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "=============================="

cd "${SLURM_SUBMIT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python prepare_lfw.py \
    --input-dir lfw/lfw-deepfunneled/lfw-deepfunneled \
    --output-dir dataset/original

