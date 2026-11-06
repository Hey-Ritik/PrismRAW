"""Start locally by default; Docker/Render explicitly bind to 0.0.0.0."""

import argparse
import os
from pathlib import Path
import uvicorn

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path.cwd() / ".cache/torch"))
    uvicorn.run("smartphone_isp.app.server:app", host=args.host, port=args.port, log_level="info")
