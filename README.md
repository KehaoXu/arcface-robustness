# ArcFace Robustness Analysis

This project evaluates how image degradation affects ArcFace verification performance.

The reusable pipeline logic lives under `face_pipeline/`, while the root-level `*.py` files are the CLI entry points used in the protocol below.

## Protocol 

1. Prepare the original `LFW` images under a normalized `dataset/original` directory with consistent preprocessing (e.g., resize).
   Interface: `prepare_dataset.py`

```powershell
python prepare_dataset.py --input-dir lfw/lfw-deepfunneled/lfw-deepfunneled --output-dir dataset/original
```

2. Build `sample_index` from `dataset/original`, keep identities with at least `10` images, and write `sample_index.csv` plus `identity_filter_summary.json`.
   Interface: `build_sample_index.py`

```powershell
python build_sample_index.py --original-dir dataset/original --output-dir results/baseline
```

3. Build a fixed `pair_manifest` from the protocol-eligible samples for all later evaluations.
   Interface: `build_pair_manifest.py`

```powershell
python build_pair_manifest.py --sample-index-path results/baseline/sample_index.csv --output-dir results/baseline
```

4. Evaluate `original`: extract embeddings, compute cosine similarity scores, compute ROC / AUC / EER, and save the fixed EER threshold.
   Interface: `evaluate_original.py`

```powershell
python evaluate_original.py --original-dir dataset/original --baseline-dir results/baseline
```

5. Generate `downsample` and `noise` images only for samples in the protocol.
   Interface: `generate_downsample_conditions.py`, `generate_noise_conditions.py`

```powershell
python generate_downsample_conditions.py --original-dir dataset/original --output-dir dataset/downsample --sample-index-path results/baseline/sample_index.csv --eligible-only
```

```powershell
python generate_noise_conditions.py --original-dir dataset/original --output-dir dataset/noise --sample-index-path results/baseline/sample_index.csv --eligible-only
```

6. Evaluate the `downsample` and `noise` condition using the same `pair_manifest` and the fixed threshold.
   Interface: `evaluate_condition.py`

```powershell
python evaluate_condition.py --condition-name downsample --condition-dir dataset/downsample --baseline-dir results/baseline --output-dir results/downsample
```

```powershell
python evaluate_condition.py --condition-name noise --condition-dir dataset/noise --baseline-dir results/baseline --output-dir results/noise
```

7. Compare conditions using fixed-threshold classification metrics, embedding drift relative to `original`, and genuine / impostor score distributions.
   Interface: `aggregate_results.py`

```powershell
python aggregate_results.py --baseline-dir results/baseline --original-dir results/baseline --downsample-dir results/downsample --noise-dir results/noise --aggregate-dir results/final
```

## Naming Conventions

- Python files use `snake_case`.
- Paths use `_path` or `_dir` suffixes.
- Collections use plural names such as `sample_records` and `pair_records`.
- Conditions use fixed names:
  - `original`
  - `downsample`
  - `noise`

After all individual steps are validated, you can optionally run the baseline end to end with `build_baseline.py`.

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
