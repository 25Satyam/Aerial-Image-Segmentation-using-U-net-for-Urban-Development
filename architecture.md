# U-Net Architecture Documentation

## Overview

U-Net was introduced by Ronneberger et al. in 2015 for biomedical image segmentation and has since become the dominant architecture for dense prediction tasks in satellite and aerial imagery.

Its key insight is the **encoder-decoder structure with skip connections** — the encoder captures semantics while the decoder recovers spatial resolution, and skip connections ensure fine-grained details are preserved.

---

## Vanilla U-Net

```
Input (B × 3 × 512 × 512)
        │
   ┌────▼────┐
   │ Enc1 64 │  ────────────────────────────────────────┐ skip4
   └────┬────┘                                          │
    MaxPool                                             │
   ┌────▼─────┐                                         │
   │ Enc2 128 │  ───────────────────────────────────┐   │ skip3
   └────┬─────┘                                     │   │
    MaxPool                                         │   │
   ┌────▼─────┐                                     │   │
   │ Enc3 256 │  ────────────────────────────────┐  │   │ skip2
   └────┬─────┘                                  │  │   │
    MaxPool                                      │  │   │
   ┌────▼─────┐                                  │  │   │
   │ Enc4 512 │  ─────────────────────────────┐  │  │   │ skip1
   └────┬─────┘                               │  │  │   │
    MaxPool                                   │  │  │   │
   ┌────▼──────┐                              │  │  │   │
   │ Bottleneck│                              │  │  │   │
   │   1024    │                              │  │  │   │
   └────┬──────┘                              │  │  │   │
    Upsample                                  │  │  │   │
   ┌────▼─────┐                               │  │  │   │
   │  Dec4 512│◄──────────────────────────────┘  │  │   │ cat
   └────┬─────┘                                  │  │   │
    Upsample                                     │  │   │
   ┌────▼─────┐                                  │  │   │
   │  Dec3 256│◄───────────────────────────────── ┘  │   │ cat
   └────┬─────┘                                      │   │
    Upsample                                         │   │
   ┌────▼─────┐                                      │   │
   │  Dec2 128│◄────────────────────────────────────── ┘   │ cat
   └────┬─────┘                                          │
    Upsample                                             │
   ┌────▼────┐                                           │
   │  Dec1 64│◄──────────────────────────────────────────┘ cat
   └────┬────┘
   1×1 Conv
        │
   Output (B × N_classes × 512 × 512)
```

---

## Key Design Decisions

### 1. Skip Connections
Skip connections concatenate encoder feature maps to corresponding decoder layers. This allows the network to combine:
- **Low-level features** (edges, textures) from early encoder layers
- **High-level semantics** (class context) from deep encoder layers

Without skip connections, fine spatial detail is lost during downsampling.

### 2. Double Convolution Blocks
Each stage uses two consecutive `3×3 Conv → BatchNorm → ReLU` operations, which:
- Doubles the receptive field compared to a single conv
- BatchNorm stabilises training and allows higher learning rates
- Same padding preserves spatial dimensions within each block

### 3. Bilinear Upsampling vs Transposed Convolution
We support both:
- **Bilinear** (default): Simpler, avoids checkerboard artefacts, followed by 1×1 conv to reduce channels
- **ConvTranspose2d**: Learnable upsampling, slightly better at recovering fine detail

### 4. Bottleneck
The deepest layer (1024 channels at 32×32 for a 512-input) captures global context about the entire scene — critical for distinguishing building rooftops from roads at the macro level.

---

## ResNet-34 Backbone Variant

For tasks with limited labelled data, we provide `UNetResNet` — a U-Net with a **pretrained ResNet-34 encoder**.

Benefits:
- ImageNet-pretrained weights provide a rich feature initialisation
- ResNet's residual connections mitigate vanishing gradients
- Typically +3–5% mIoU improvement over vanilla U-Net with same data

The ResNet encoder outputs 5 skip feature maps at strides {2, 4, 8, 16, 32} which feed into the decoder.

---

## Input/Output

| Property | Value |
|---|---|
| Input | 3-channel RGB, any size divisible by 16 |
| Normalisation | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |
| Output | Raw logits, shape (B, N_classes, H, W) |
| Prediction | `argmax(logits, dim=1)` → class indices |

---

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. https://arxiv.org/abs/1505.04597
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Iglovikov, V. & Shvets, A. (2018). TernausNet: U-Net with VGG11 Encoder.
