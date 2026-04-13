#!/bin/bash
#SBATCH --job-name=face-gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
## If your cluster requires these, uncomment and fill them in:
## #SBATCH --account=your_account
## #SBATCH --qos=your_qos

set -euo pipefail

mkdir -p logs

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
MODE="${MODE:-evaluation}"
CONDA_ENV="${CONDA_ENV:-resshift}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========== Job Info =========="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Mode: ${MODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "=============================="

cd "${PROJECT_ROOT}"

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/11.8 || true
fi

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

run_evaluation() {
    ORIGINAL_DIR="${ORIGINAL_DIR:-dataset/original}"
    DOWNSAMPLE_DIR="${DOWNSAMPLE_DIR:-dataset/downsample}"
    NOISE_DIR="${NOISE_DIR:-dataset/noise}"
    DOWNSAMPLE_RESSHIFT_DIR="${DOWNSAMPLE_RESSHIFT_DIR:-dataset/downsample_resshift}"
    NOISE_RESSHIFT_DIR="${NOISE_RESSHIFT_DIR:-dataset/noise_resshift}"
    RESULTS_ROOT="${RESULTS_ROOT:-results}"
    MODEL_NAME="${MODEL_NAME:-buffalo_l}"

    echo "[evaluation] Running baseline and all condition evaluations"

    "${PYTHON_BIN}" run_baseline.py \
        --original-dir "${ORIGINAL_DIR}" \
        --output-dir "${RESULTS_ROOT}/baseline" \
        --model-name "${MODEL_NAME}"

    "${PYTHON_BIN}" run_condition_evaluation.py \
        --condition-name downsample \
        --condition-dir "${DOWNSAMPLE_DIR}" \
        --baseline-dir "${RESULTS_ROOT}/baseline" \
        --output-dir "${RESULTS_ROOT}/downsample" \
        --model-name "${MODEL_NAME}"

    "${PYTHON_BIN}" run_condition_evaluation.py \
        --condition-name noise \
        --condition-dir "${NOISE_DIR}" \
        --baseline-dir "${RESULTS_ROOT}/baseline" \
        --output-dir "${RESULTS_ROOT}/noise" \
        --model-name "${MODEL_NAME}"

    "${PYTHON_BIN}" run_condition_evaluation.py \
        --condition-name downsample_resshift \
        --condition-dir "${DOWNSAMPLE_RESSHIFT_DIR}" \
        --baseline-dir "${RESULTS_ROOT}/baseline" \
        --output-dir "${RESULTS_ROOT}/downsample_resshift" \
        --model-name "${MODEL_NAME}"

    "${PYTHON_BIN}" run_condition_evaluation.py \
        --condition-name noise_resshift \
        --condition-dir "${NOISE_RESSHIFT_DIR}" \
        --baseline-dir "${RESULTS_ROOT}/baseline" \
        --output-dir "${RESULTS_ROOT}/noise_resshift" \
        --model-name "${MODEL_NAME}"

    "${PYTHON_BIN}" run_full_evaluation.py \
        --baseline-dir "${RESULTS_ROOT}/baseline" \
        --original-dir "${RESULTS_ROOT}/baseline" \
        --downsample-dir "${RESULTS_ROOT}/downsample" \
        --noise-dir "${RESULTS_ROOT}/noise" \
        --downsample-resshift-dir "${RESULTS_ROOT}/downsample_resshift" \
        --noise-resshift-dir "${RESULTS_ROOT}/noise_resshift" \
        --aggregate-dir "${RESULTS_ROOT}/final"
}

run_prepare_lfw() {
    INPUT_DIR="${INPUT_DIR:-lfw/lfw-deepfunneled/lfw-deepfunneled}"
    OUTPUT_DIR="${OUTPUT_DIR:-dataset/original}"
    WIDTH="${WIDTH:-512}"
    HEIGHT="${HEIGHT:-512}"
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-.png}"

    echo "[prepare_lfw] Preparing normalized LFW images"
    echo "[prepare_lfw] input=${INPUT_DIR} output=${OUTPUT_DIR} size=${WIDTH}x${HEIGHT} suffix=${OUTPUT_SUFFIX}"

    "${PYTHON_BIN}" prepare_lfw.py \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --width "${WIDTH}" \
        --height "${HEIGHT}" \
        --output-suffix "${OUTPUT_SUFFIX}"
}

run_resshift_infer() {
    IN_PATH="${IN_PATH:-../dataset/downsample}"
    OUT_PATH="${OUT_PATH:-../dataset/downsample_resshift}"
    TASK="${TASK:-faceir}"
    SCALE="${SCALE:-1}"
    VERSION="${VERSION:-v3}"
    BS="${BS:-1}"
    CHOP_SIZE="${CHOP_SIZE:-256}"
    CHOP_STRIDE="${CHOP_STRIDE:--1}"
    MASK_PATH="${MASK_PATH:-}"

    echo "[resshift_infer] Running ResShift inference"

    cd "${PROJECT_ROOT}/ResShift"

    CMD=(
        "${PYTHON_BIN}" inference_resshift.py
        -i "${IN_PATH}"
        -o "${OUT_PATH}"
        --task "${TASK}"
        --scale "${SCALE}"
        --version "${VERSION}"
        --bs "${BS}"
        --chop_size "${CHOP_SIZE}"
        --chop_stride "${CHOP_STRIDE}"
    )

    if [ -n "${MASK_PATH}" ]; then
        CMD+=(--mask_path "${MASK_PATH}")
    fi

    "${CMD[@]}"
}

run_resshift_train() {
    CFG_PATH="${CFG_PATH:-configs/faceir_gfpgan512_lpips.yaml}"
    SAVE_DIR="${SAVE_DIR:-train_logs}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"

    echo "[resshift_train] Running multi-GPU ResShift training"
    echo "[resshift_train] cfg=${CFG_PATH} save_dir=${SAVE_DIR} gpus=${GPUS_PER_NODE}"

    cd "${PROJECT_ROOT}/ResShift"

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${GPUS_PER_NODE}" \
        main.py \
        --cfg_path "${CFG_PATH}" \
        --save_dir "${SAVE_DIR}"
}

case "${MODE}" in
    prepare_lfw)
        run_prepare_lfw
        ;;
    evaluation)
        run_evaluation
        ;;
    resshift_infer)
        run_resshift_infer
        ;;
    resshift_train)
        run_resshift_train
        ;;
    *)
        echo "Unsupported MODE: ${MODE}"
        echo "Supported values: prepare_lfw, evaluation, resshift_infer, resshift_train"
        exit 1
        ;;
esac

echo "Finished at: $(date)"
