---
title:  Understanding GPU Computational Capabilities
event:  CSC Summer School in High-Performance Computing 2025
author: Hossein Firooz (FCAI / Aalto University)
lang:   en
---

# GPUs on LUMI-G

:::::: {.columns}
::: {.column width="60%"}
![](img/lumi-g.svg){.center width=100%}

<small>LUMI-G Node.</small>
:::
::: {.column width="40%"}
![](img/amd-mi250.avif){.center width=60%}

<small>Single AMD MI250X GPU</small>
:::
::::::

# FLOPS (Floating point operations per second)

- Performance is often measured in FLOPS (FLOP/s)
- FLOP count measure the number of arithmetic operations a model performs
  - Commonly used to estimate compute cost of training/inference
- Training ML Models = Forward pass + Backward pass

# AMD MI250X GPU Characteristics
- Compute Power [(Link)](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html)
    - Peak FP64 Performance: 47.9 TFLOPS
    - Peak FP32 Performance: 47.9 TFLOPS
    - Peak FP16 Performance: 383 TFLOPS

- Memory
    - 128 GB HBM2e (64 GB per GCD)

- These numbers are for 2 GCDs with 220 CUs
    - `--gpus-per-task=1` gives you one GCD

# Peak vs Max-Achievable FLOPS

:::::: {.columns}
::: {.column width="50%"}
<center>
![](img/maf-flops.png){.center width=80%}
<small>Picture from [AMD](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)</small>
</center>

:::
::: {.column width="45%"}
- Peak performance is calculated based on the hardware characteristics
  - <small>$\text{FLOPS} = \text{Cores} \times \text{Ops/Cycle} \times \text{Clock}$</small>
- Memory Bandwidth Limits, Underutilization, Load Imbalance, etc.
- Usually **40–70% of Peak FLOPS** in practice
:::
::::::

# ML Parameters vs FLOP count

- **No. Parameters** are static — they define the model size
- **FLOP count** depends on data:
    - Input size
    - Number of filters, etc
    - Dataset size (for total cost)
- A model with few parameters can still have high FLOP count if it processes high-resolution inputs

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
    - Forward FLOP count per image: ~11.5 GFLOP
    - Backward FLOP = 2x forward pass [(Link)](https://epoch.ai/blog/backward-forward-FLOP-ratio)
    - Training FLOP per image: ~34.5 GFLOP

```python
import fvcore
model = resnet152().to(device)
model.eval()

input = torch.randn(1, 3, 224, 224).to(device)

flops = fvcore.nn.FlopCountAnalysis(model, input)
total_params = sum(p.numel() for p in model.parameters())
```

# VRAM Estimate: ResNet-152 + CIFAR-100 (224x224)

| Component          | FP32 (approx) |
|-------------------|---------------|
| Parameters         | 240 MB        |
| Adam  Optimizer    | 480 MB        |
| Gradients          | 240 MB        |
| Activations*       | ~180 MB       |
| Overhead           | ~1 GB         |
| **Total**          | ~2 GB         |

- For `batch_size=128` VRAM is ~24Gb .

# Example: ResNet-152 with CIFAR-100

- **Per-Image Total FLOP Count:**
$3 \times 11.5\ \text{GFLOP} = 34.5\ \text{GFLOP\ per\ image}$

- **Total Epoch FLOP Count:**
$34.5\ \text{GFLOP} \times 50000 = 1.725\ \text{PFLOP}$


- **Usable Throughput for MI250X GPU (Assuming 40% Efficiency):**
$0.40 \times 47.9\ \text{TFLOPS} = 19.2\ \text{TFLOPS}$

- **Usable Throughput for Single GCD (2 GCD per GPU):**
$\frac{19.2\ \text{TFLOPS}}{2} = 9.6\ \text{TFLOPS}$

- **Estimated Epoch Time:**
$\frac{1.725\ \text{PFLOP}}{9.6\ \text{TFLOPS}} \approx 180\ \text{seconds}$


# Real-world performance

- Performance highly depends on Code implementation, I/O, etc.
- That's why profiling matters:
    - Visualize computation, memory, communication.
    - Identify bottlenecks early.
    - Optimize GPU usage efficiently

# Key Takeaways

- FLOP count depends also on input, not just model size
- VRAM usage is dominated by activations, especially in deep models
  - We have control over the `batch_size`
- Mixed precision and parallelism help reach closer to max achievable FLOPS
- Always measure real-time training performance to understand bottlenecks
