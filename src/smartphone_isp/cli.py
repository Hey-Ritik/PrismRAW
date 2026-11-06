"""Project utilities; inference commands will follow dataset validation."""

import argparse
import importlib.util
import platform


def main():
    parser = argparse.ArgumentParser(description="Smartphone ISP project utilities")
    parser.add_argument("command", choices=["check"])
    parser.parse_args()
    print(f"Python: {platform.python_version()}")
    for name in ("numpy", "cv2", "rawpy", "scipy", "torch", "gradio", "skimage", "lpips"):
        status = "available" if importlib.util.find_spec(name) else "not installed"
        print(f"{name}: {status}")
    print("Status: scaffold; no trained checkpoint or complete inference pipeline yet.")


if __name__ == "__main__":
    main()
