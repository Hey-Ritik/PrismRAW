# Architecture

Classical array inference, paired training, learned-only packed-Bayer reconstruction, and checkpoint-based hybrid inference are implemented. Direct DNG decoding and deployment optimization remain pending. Locally generated pilot weights are Git-ignored; classical remains the strongest pilot method.

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
    B --> H[Packed Bayer + camera metadata neural reconstruction]
    H --> J[Common display rendering]
    G --> I[Comparison UI and evaluation]
    J --> I
```

All methods must produce the same target RGB encoding and spatial dimensions. Learned-only and hybrid paths have distinct model/input contracts. Input normalization shared between methods is not an enhancement baseline.
