# Implemented architecture

```mermaid
flowchart TD
    A[DNG or Bayer array + metadata] --> B[Explicit sensor decoding and normalization]
    B --> C[Noise / illumination estimates]
    B --> D[White balance + bilinear demosaicing + camera colour transform]
    D --> E[Edge-preserving bilateral denoising]
    E --> F[Conditioned residual U-Net / FP32 or FP16]
    E --> G[Adaptive tone mapping + sRGB + controlled sharpening]
    F --> G
    C --> F
    C --> G
    B --> H[Packed Bayer + camera features]
    H --> I[Learned-only resize U-Net RGB reconstruction]
    I --> G
    G --> J[Four-way UI, intermediates, downloads and optional metrics]
    K[Matching clean RAW or registered RGB reference] --> J
```

The common display renderer makes output encoding comparable; learned-only does not use classical demosaicing, white-balance multiplication, matrix multiplication or denoising before its network. Training/evaluation share equal RAW context and crop predictions to central targets. Full app inference returns the selected even-sized sensor crop.

Data preparation freezes physical-scene assignments before patching. Training chooses checkpoints on validation. The optimization step selects residual blending on validation and exports FP16. Fresh-scene quality, ablations/stress tests, isolated CPU benchmarks and registration checks feed measured reports. The app consumes only final weights and small sample crops, not the full training set. Public hosting is separate from this verified local architecture.
