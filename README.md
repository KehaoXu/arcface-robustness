# ArcFace Robustness Analysis

This project evaluates how image degradation affects ArcFace verification performance.

The reusable pipeline logic lives under `face_pipeline/`, while the root-level `*.py` files are the CLI entry points used in the protocol below.

## Protocol 

1. Prepare the original `LFW` images under a normalized `/scratch/peirong/kxu56/face/dataset/original` directory with consistent preprocessing (e.g., resize).
   Interface: `prepare_dataset.py`

```powershell
python prepare_dataset.py --input-dir lfw/lfw-deepfunneled/lfw-deepfunneled --output-dir /scratch/peirong/kxu56/face/dataset/original
```

2. Build `sample_index` from `/scratch/peirong/kxu56/face/dataset/original`, keep identities with at least `10` images, then build a fixed `pair_manifest` for all later evaluations.
   Interface: `build_sample_pair.py`

```powershell
python build_sample_pair.py --original-dir /scratch/peirong/kxu56/face/dataset/original --output-dir results/baseline
```

3. Evaluate `original`: extract embeddings, compute cosine similarity scores, compute ROC / AUC / EER, and save the fixed EER threshold.
   Interface: `evaluate_original.py`

```powershell
python evaluate_original.py --original-dir /scratch/peirong/kxu56/face/dataset/original --baseline-dir results/baseline
```

4. Generate `downsample` and `noise` images only for samples in the protocol.
   Interface: `generate_downsample.py`, `generate_noise.py`

```powershell
python generate_downsample.py \
    --downsample-scale 0.5 \
    --original-dir /scratch/peirong/kxu56/face/dataset/original \
    --output-dir /scratch/peirong/kxu56/face/dataset/downsample \
    --sample-index-path results/pair/sample_index.csv \
    --eligible-only
```

```powershell
python generate_noise.py \
    --gaussian-sigma 16.0 \
    --original-dir /scratch/peirong/kxu56/dataset/face/original \
    --output-dir /scratch/peirong/kxu56/dataset/face/noise \
    --sample-index-path results/pair/sample_index.csv \
    --eligible-only
```

```powershell
python inference_resshift.py \
    --inference-steps 4 \
    -i /scratch/peirong/kxu56/face/dataset/noise \
    -o /scratch/peirong/kxu56/face/dataset/noise_re \
    --task faceir \
    --scale 1 \
    --bs 1
```

Useful hyperparameters:

- `generate_downsample.py --downsample-scale 0.5`
- `generate_noise.py --gaussian-sigma 16.0`
- `ResShift/inference_resshift.py --inference-steps 4`

5. Evaluate the `downsample` and `noise` condition using the same `pair_manifest` and the fixed threshold.
   Interface: `evaluate_condition.py`

```powershell
python evaluate_condition.py --condition-name downsample --condition-dir /scratch/peirong/kxu56/face/dataset/downsample --baseline-dir results/baseline --output-dir results/downsample
```

```powershell
python evaluate_condition.py --condition-name noise --condition-dir /scratch/peirong/kxu56/face/dataset/noise --baseline-dir results/baseline --output-dir results/noise
```

6. Compare conditions using fixed-threshold classification metrics, embedding drift relative to `original`, and genuine / impostor score distributions.
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
