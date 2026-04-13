# ArcFace Robustness Analysis

This project evaluates how image degradation affects ArcFace verification performance.

## Protocol

1. Prepare the original `LFW` images as a normalized `dataset/original` directory.
2. Build the baseline on `original` only:
   - create `sample_index`
   - keep identities with at least `10` images
   - generate a fixed `pair_manifest`
   - extract embeddings
   - compute ROC / AUC / EER
   - save the fixed threshold
3. Generate `downsample` and `noise` for the protocol samples you want to evaluate.
4. Evaluate `downsample` and `noise` with that fixed threshold.
5. Compare conditions using:
   - fixed-threshold classification metrics
   - embedding drift relative to `original`
   - genuine / impostor score distributions

## Canonical Python Entry Points

- `prepare_lfw.py`
  Resizes and converts the original LFW dataset into a normalized directory structure.

- `generate_image_conditions.py`
  Generates `downsample` and `noise` image conditions from `dataset/original`. The `downsample` quality is controlled by `downsample_scale`, and the `noise` quality is controlled by `gaussian_sigma`. It can optionally read `sample_index.csv` and only process protocol-eligible samples.

- `run_baseline.py`
  Builds the baseline protocol and fixed threshold from `dataset/original`.

- `run_condition_evaluation.py`
  Evaluates one condition against the baseline protocol and threshold.

- `run_full_evaluation.py`
  Aggregates all condition outputs and generates the final comparison figures.

## Naming Conventions

- Python files use `snake_case`.
- Paths use `_path` or `_dir` suffixes.
- Collections use plural names such as `sample_records` and `pair_records`.
- Conditions use fixed names:
  - `original`
  - `downsample`
  - `noise`

## Example Commands

### Step 1. Prepare normalized LFW images

```powershell
python prepare_lfw.py --input-dir lfw/lfw-deepfunneled/lfw-deepfunneled --output-dir dataset/original
```

### Step 2. Build the baseline on original images

```powershell
python run_baseline.py --original-dir dataset/original --output-dir results/baseline
```

### Step 3. Generate degraded image conditions

This filtered form only processes samples already marked as protocol-eligible in `results/baseline/sample_index.csv`.

```powershell
python generate_image_conditions.py --original-dir dataset/original --downsample-dir dataset/downsample --noise-dir dataset/noise --sample-index-path results/baseline/sample_index.csv --eligible-only
```

### Step 4. Evaluate the `downsample` condition

```powershell
python run_condition_evaluation.py --condition-name downsample --condition-dir dataset/downsample --baseline-dir results/baseline --output-dir results/downsample
```

### Step 5. Evaluate the `noise` condition

```powershell
python run_condition_evaluation.py --condition-name noise --condition-dir dataset/noise --baseline-dir results/baseline --output-dir results/noise
```

### Step 6. Aggregate all results and generate figures

```powershell
python run_full_evaluation.py --baseline-dir results/baseline --original-dir results/baseline --downsample-dir results/downsample --noise-dir results/noise --aggregate-dir results/final
```

## Outputs

The pipeline writes structured CSV / JSON artifacts such as:

- `sample_index.csv`
- `identity_filter_summary.json`
- `pair_manifest.csv`
- `threshold_summary.json`
- `embedding_results.csv`
- `pair_scores_with_predictions.csv`
- `embedding_drift.csv`
- `metrics_summary.json`
- `distribution_summary.csv`
- `results/final/figures/*.png`
