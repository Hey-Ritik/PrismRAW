# Architecture

This diagram is the intended architecture; only foundational utilities and the untrained refinement model currently exist.

```mermaid
flowchart TD
    A[RAW + metadata] --> B[Decode and validated normalization]
    B --> C[Noise / illumination estimates]
    B --> D[White balance + demosaicing + camera colour transform]
    D --> E[Classical edge-preserving denoiser]
    D --> F[Learned residual refinement]
    C --> F
    E --> G[Tone mapping + output encoding + sharpening]
    F --> G
    B --> H[Packed Bayer learned-only baseline]
    G --> I[Comparison UI and evaluation]
    H --> I
```

All methods must produce the same target RGB encoding and spatial dimensions. Learned-only and hybrid paths have distinct model/input contracts. Input normalization shared between methods is not an enhancement baseline.
