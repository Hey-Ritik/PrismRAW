# Learned-model experiments and studio improvements

Three isolated learned-model fine-tuning trials were completed: stronger detail loss with Bayer-safe augmentation, lower-learning-rate refinement without augmentation, and a broader-data trial. None replaced the production checkpoint.

The broader-data trial adds two pinned, checksum-verified high-ISO paired captures from physical training scenes 003 and 004. Training increases from 192 to 256 patches. All scene assignments remain unchanged; all 64 prepared validation pairs are byte-for-byte equivalent as numeric arrays. The original test report remains historical and was not recomputed to select a candidate.

Candidates were compared on the same validation scenes 001/002. Promotion required lower LPIPS, no decrease in PSNR or SSIM, and no increase in colour error relative to the deployed learned model. The broader-data candidate increased PSNR from 24.4154 to 25.3538 dB and SSIM from 0.9225 to 0.9259, but Delta E worsened from 10.6538 to 11.6093 and LPIPS from 0.2745 to 0.2798. It was rejected. Higher PSNR alone is not evidence of an overall quality improvement.

Full configs, checkpoint hashes, sample counts and measured results are in backend/reports/learned_improvement_trials.json. Capture URLs and hashes are in backend/reports/learned_data_expansion.json. Candidate weights and training arrays remain local and Git-ignored. Production weights, memory requirements and the existing quality baseline remain unchanged.

## Reproduce from backend/

The original pinned dataset and prepared final manifest must be present first. See the existing data preparation instructions.

```text
python scripts/download_detail_captures.py
python scripts/prepare_training_data.py --index data/raw/sidd-cc-subset/detail-index.json --output data/processed/detail --patches-per-scene 32 --seed 42 --existing-splits configs/final-splits.json --input-context 4
python -m smartphone_isp.training.train --config configs/learned-detail-candidate.yaml
python -m smartphone_isp.training.train --config configs/learned-detail-refine.yaml
python -m smartphone_isp.training.train --config configs/learned-detail-expanded.yaml
python scripts/evaluate_learned_candidates.py
python scripts/write_failure_analysis.py
```

Resume validation checks both the previous manifest hash and physical-scene assignments before allowing an expanded training manifest. A changed validation/test assignment is rejected. Training checkpoints are selected using validation linear-RGB L1; the separate promotion gate evaluates displayed-image quality.

## Studio behavior

Health checks use a 12-second timeout and at most six attempts, with 3-second delays between attempts. A visible Retry connection button appears after exhaustion. User processing and report requests first check readiness, including after an idle shutdown. Only health GETs are automatically retried; a failed image-processing POST is never automatically resent. There is no background keep-alive timer.

Three highlights show highest PSNR, lowest Delta E and fastest processing for the current capture. Ties are preserved. Without a clean reference, quality highlights say Needs reference while measured latency remains available. Single-run latency highlights are not benchmark rankings.

The failure viewer loads two measured held-out examples via /api/failures and /failure-examples. They are retrospectively selected examples from the frozen report, not a fresh evaluation or model-selection set. Labels distinguish reference, classical, learned and hybrid. The written report includes the rejected follow-up training trials.
