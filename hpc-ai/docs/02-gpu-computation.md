---
title:  Undernstading GPU computational capabilities
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# GPUs on LUMI-G

<div class="column"  style="width:50%"; text-align: center;">
  ![](img/lumi-g.svg){width=20%}
  - <small>LUMI-G Node.</small>
</div>
<div class="column"  style="width:50%">
  ![](img/amd-mi250.avif){width=60%}
  - <small>Single AMD MI250 GPU</small>
</div>

# GPU Charachteristics
- Computer Power
    - Peak FP64 Performance: 47.9 TFLOPs (per GPU)
    - Peak FP32 Performance: 95.7 TFLOPs
    - Peak FP16 Performance: 383 TFLOPs

- Memory
- 128 GB HBM2e (64 GB per GCD)
- 3.2 TB/s total memory bandwidth


# FLOPs (Floating Point Operations)

- FLOPs measure how many arithmetic operations a model performs.
- Commonly used to estimate compute cost of training/inference.
- Training ML Models = 2 × FLOPs (forward + backward).


# ML Parameters vs FLOPs

- **No. Parameters** are static — they define the model size.
- **FLOPs** depend on:
    - Input image size
    - Number of filters
    - Spatial dimensions
    - Batch size (for total cost)

- A model with few parameters can still have high FLOPs if it processes high-resolution inputs.

# VRAM Usage Breakdown

| Component            | Description                                  |
|---------------------|----------------------------------------------|
| Model Parameters     | Static memory for weights                    |
| Gradients            | Stored during backpropagation                |
| Optimizer States     | e.g., momentum/Adam stats                    |
| Activations          | Largest source — intermediate tensors        |
| Framework Overhead   | Memory allocator, workspace, caching         |

- Peak VRAM usage is dominated by activations in deep CNNs.

# Example: ResNet-152 with CIFAR-100

- **Model Info:**
    - Parameters: ~60.2M
    - FLOPs per image (forward): ~11.5 GFLOPs
    - Training FLOPs per image: ~23 GFLOPs

```python
import fvcore
model = resnet152().to(device)
model.eval()

input = torch.randn(1, 3, 224, 224).to(device)

flops = fvcore.nn.FlopCountAnalysis(model, input)
```

# VRAM Estimate: ResNet-152 + CIFAR-100 (224x224)

| Component          | FP32 (approx) |
|-------------------|---------------|
| Parameters         | 240 MB        |
| Adam  Optimizer    | 480 MB        |
| Gradients          | 240 MB        |
| Activations        | 12–14 GB      |
| Overhead           | ~1 GB         |
| **Total**          | ~13–15 GB     |


# Example: ResNet-152 with CIFAR-100

- **Convolution FLOPs Estimate**  
$\text{FLOPs} = 2 \times C_{\text{in}} \times K^2 \times H \times W \times C_{\text{out}}$

- **Per-Image Total FLOPs**  
\[
11.5 times 2\,\text{GFLOPs} = \boxed{23\,\text{GFLOPs per image}}
\]

- **Usable GPU Throughput (Assuming 35% Efficiency)**  
\[
\text{Usable Throughput} = 0.35 \times 95.7 = \boxed{33.5\,\text{TFLOPs/s}}
\]

- **Total Epoch FLOPs**  
\[
\text{Epoch FLOPs} 23\,\text{GFLOPs} \times 50k= 1.15\,\text{PFLOPs}
\]

# Key Takeaways

- FLOPs are a function of image size, not just model size.
- VRAM usage is dominated by activations, especially in deep models.
- Mixed precision and parallelism help reach closer to max achievable FLOPs.
- Always measure real-time training performance to understand bottlenecks.
