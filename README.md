# ArcFace Robustness Analysis

This project evaluates how image degradation and ResShift restoration affect ArcFace verification performance.

## Protocol

1. Prepare the original `LFW` images as a normalized `dataset/original` directory.
2. Build the baseline on `original` only:
   - create `sample_index`
   - keep identities with at least `10` images
   - generate a fixed `pair_manifest`
   - extract embeddings
   - compute ROC / AUC / EER
   - save the fixed threshold
3. Evaluate `downsample`, `noise`, `downsample_resshift`, and `noise_resshift` with that fixed threshold.
4. Compare conditions using:
   - fixed-threshold classification metrics
   - embedding drift relative to `original`
   - genuine / impostor score distributions

## Canonical Python Entry Points

- `prepare_lfw.py`
  Resizes and converts the original LFW dataset into a normalized directory structure.

- `generate_image_conditions.py`
  Migrates the old notebook preprocessing into a script that generates `downsample` and `noise`, and can normalize externally restored `downsample_resshift` and `noise_resshift` directories into the expected layout.

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
  - `downsample_resshift`
  - `noise_resshift`

## Example Commands

```powershell
python prepare_lfw.py --input-dir lfw/lfw-deepfunneled/lfw-deepfunneled --output-dir dataset/original
python generate_image_conditions.py --original-dir dataset/original --downsample-dir dataset/downsample --noise-dir dataset/noise --downsample-resshift-source-dir path/to/resshift_downsample_output --noise-resshift-source-dir path/to/resshift_noise_output --downsample-resshift-dir dataset/downsample_resshift --noise-resshift-dir dataset/noise_resshift
python run_baseline.py --original-dir dataset/original --output-dir results/baseline
python run_condition_evaluation.py --condition-name downsample --condition-dir dataset/downsample --baseline-dir results/baseline --output-dir results/downsample
python run_condition_evaluation.py --condition-name noise --condition-dir dataset/noise --baseline-dir results/baseline --output-dir results/noise
python run_condition_evaluation.py --condition-name downsample_resshift --condition-dir dataset/downsample_resshift --baseline-dir results/baseline --output-dir results/downsample_resshift
python run_condition_evaluation.py --condition-name noise_resshift --condition-dir dataset/noise_resshift --baseline-dir results/baseline --output-dir results/noise_resshift
python run_full_evaluation.py --baseline-dir results/baseline --original-dir results/baseline --downsample-dir results/downsample --noise-dir results/noise --downsample-resshift-dir results/downsample_resshift --noise-resshift-dir results/noise_resshift --aggregate-dir results/final
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
