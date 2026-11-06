"""Small explicit runtime settings shared by local and hosted inference."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RuntimeSettings:
    max_crop: int
    threads: int
    hosted: bool


def runtime_settings():
    crop = int(os.environ.get("PRISMRAW_MAX_CROP", "512"))
    threads = int(os.environ.get("PRISMRAW_TORCH_THREADS", "2"))
    if crop < 64 or crop > 512 or crop % 2:
        raise ValueError("PRISMRAW_MAX_CROP must be even and between 64 and 512")
    if not 1 <= threads <= 8:
        raise ValueError("PRISMRAW_TORCH_THREADS must be between 1 and 8")
    hosted = os.environ.get("PRISMRAW_HOSTED", "").lower() in {"1", "true"}
    return RuntimeSettings(crop, threads, hosted)
