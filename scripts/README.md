# Experiment entry points

Implemented entry points:

1. `download_sidd_subset.py`: pinned, verified paired RAW downloads.
2. `prepare_training_data.py`: scene-disjoint linear RGB / packed Bayer pairs.
3. `python -m smartphone_isp.training.train --config ...`: separate hybrid or learned-only training.
4. `python -m smartphone_isp.evaluation.pilot`: shared pilot comparison, optionally with `--learned-checkpoint`.

Legacy single-sample utilities remain available for smoke checks. See the README for complete commands and `configs/pilot-splits.json` for frozen exploratory holdouts. Optimized export and target-device benchmarking remain pending.

Use package modules for reusable logic; keep scripts as thin entry points.
