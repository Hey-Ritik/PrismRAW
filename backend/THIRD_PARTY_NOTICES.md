# Third-party data and software

This project uses SIDD-CC, a research reprocessing of SIDD, for paired sensor data. Dataset source, pinned revision, file URLs, and SHA-256 hashes are recorded by the downloader. Refer to [SIDD](https://abdokamel.github.io/sidd/) and the [SIDD-CC dataset card](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC) for terms and citations. Small demonstration crops in a release retain this attribution; full data is not committed to Git.

The approximate single-illuminant rendering convention was informed by the authors' MIT-licensed [simple-camera-pipeline](https://github.com/AbdoKamel/simple-camera-pipeline). This implementation is not a full DNG profile processor or exact reproduction of that reference renderer.

LPIPS uses the [PerceptualSimilarity implementation](https://github.com/richzhang/PerceptualSimilarity), LPIPS v0.1 SqueezeNet calibration weights, and the torchvision pretrained ImageNet SqueezeNet backbone. See those projects for software and model license terms. It is reported as LPIPS-Squeeze, not AlexNet LPIPS.

Runtime libraries include NumPy, SciPy, Pillow, PyTorch, torchvision, scikit-image, rawpy/LibRaw, FastAPI, Uvicorn, python-multipart, psutil, and PyYAML. Their licenses remain with their authors. No external image-generation or cloud inference API is used by the local app.
