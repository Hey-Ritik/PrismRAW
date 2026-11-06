"""Start the local demo without exposing it to the public internet."""

import argparse
import os
from pathlib import Path
import uvicorn

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()
os.environ.setdefault("TORCH_HOME", str(Path.cwd() / ".cache/torch"))
uvicorn.run("smartphone_isp.app.server:app", host=args.host, port=args.port, log_level="info")
