# Final Submission Checklist — NOCTICS / T47 / P2

## Identity and file integrity

- [ ] Team confirms NOCTICS, T47, Section D and exact problem title.
- [ ] Ritik Singh is paired with USN 1RUA24CSE0371.
- [ ] Rishab Patil is paired with USN 1RUA24CSE0365.
- [ ] Original evaluation is [backend/reports/eval_P2_T47.md](../../backend/reports/eval_P2_T47.md); the separate technical report is [Technical_Solution_Report_P2_T47.md](Technical_Solution_Report_P2_T47.md).
- [ ] First line is exactly: 1RUA24CSE0371,1RUA24CSE0365
- [ ] Preserve the official template's heading sequence and numbering.
- [ ] Preserve the figures/ subfolder alongside the Markdown report.
- [ ] Resolve STUDENT-1 through STUDENT-10 and Figure 2–4 placeholders.
- [ ] Check Mermaid rendering in the faculty's submission viewer.

## Evidence and claims

- [ ] Match quality values to backend/reports/final_results.json.
- [ ] Match current isolated pipeline latency to backend/reports/latency_optimization.json.
- [ ] Keep historical benchmark.json timings separate.
- [ ] Keep the 641.19 ms complete-comparison measurement separate from standalone hybrid latency.
- [ ] State 64 target patches from two physical test scenes and approximate clean-RAW reference rendering.
- [ ] State latency input dimensions, two CPU threads, warmups/runs and excluded operations.
- [ ] Include Delta E and distinguish checkpoint size, isolated process RSS and full-application RAM.
- [ ] Do not claim full-resolution mobile real-time performance, universal superiority or guaranteed hosted memory fit.
- [ ] Retain the non-adaptation ablation's exposure/reference limitation.
- [ ] Identify rejected training experiments as validation-only; production weights were retained.
- [ ] Mark retrospective failure examples as such.
- [ ] Keep dataset, LPIPS and software attribution.
- [ ] If a final test-count claim is required, attach a dated run log; do not infer coverage percentages from counts.
- [ ] Identify the reviewed Git snapshot; do not invent original experiment commit hashes.

## Deployment and quickstart

- [ ] Verify the frontend and backend links near the actual submission time.
- [ ] Retain the user's deployment statement as user-verified, without adding unspecified passed cases.
- [ ] Add actual deployment observations/evidence if available.
- [ ] Do not treat a health response as proof of all inference, uploads, downloads or deployed speed.
- [ ] If clean-install reproducibility is claimed, perform and record it rather than assuming it from these instructions.
- [ ] Do not confuse Vercel Root Directory frontend with the Render Docker context backend.
- [ ] Understand temporary result storage and download important results promptly.
- [ ] Note that the live evaluation endpoint currently exposes historical timing records.

## Authorship and faculty policy

- [ ] Include the separate AI Assistance Declaration.
- [ ] State that Codex assisted implementation, debugging, testing, optimization, documentation and report preparation.
- [ ] Identify retained generated report passages and diagrams according to faculty rules.
- [ ] Add only genuine member contributions and observations.
- [ ] Obtain the written 30% policy and its measurement/allowed-use definitions.
- [ ] Do not claim an AI percentage or policy compliance without a defensible basis.
- [ ] Both members review the final report and acknowledge the declaration as required.

## Existing discrepancies — not silently corrected in the repository

- backend/docs/design_document.md and backend/docs/architecture.md do not exist; use docs/design.md and docs/architecture.md.
- Historical benchmark numbers are retained. The original evaluation and this technical report now record the public Vercel/Render URLs; other historical project documents may still contain pending-hosting language.
- Original experiment commit hashes are not consistently embedded in the saved reports.
- verification.json records an earlier 27-test run, while later development runs reported 41 tests.
- Backend deployment revision and final deployed performance need their own evidence.

This submission package remains an AI-assisted draft requiring genuine student input. Documentation-only deployment corrections do not change application code, weights, numeric evaluation results or deployment settings. Commit/push of these documents is authorized separately from publishing or redeploying the application.
