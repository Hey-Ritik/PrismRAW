# Local camera-lab application

`src/smartphone_isp/app/server.py` exposes a FastAPI service; `engine.py` runs bounded inference and packages results. `app/static/` is the responsive browser UI. Launch with `./run_local.ps1` and open http://127.0.0.1:7860 after model/LPIPS warmup.

The app has sample scenes, DNG/array uploads, four output modes, optional reference metrics, stage previews, zoom, adaptive controls, measured evaluation summaries and result ZIP downloads. Uploaded array metadata is required; no JPEG-as-RAW or unreferenced quality scores are invented. See docs/deployment.md for limits, privacy, asset requirements and container packaging.

Studio v2 provides thumbnail selection, a large split comparison viewer with pointer/keyboard controls, single view, selected-output summaries and expandable technical details. Sample previews in `static/samples` are generated from the attributed SIDD-CC clean demo captures by `scripts/prepare_ui_thumbnails.py`, not stock placeholders. `scripts/check_ui.py` verifies the full local interaction flow and responsive layout. Processing APIs and trained model outputs are unchanged.
