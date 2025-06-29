---
title:  "Scaling PyTorch Models: Single vs Multi-GPU Training and Techniques"
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Agenda

- Single-GPU
- Multi-GPU
- Data Parallelism
  - Basic Data Parallel (DP)
  - ZeRO Data Parallel (DeepSpeed)
  - Distributed Data Parallel (DDP)
- Model Parallelism
  - Tensor Parallelism
  - Pipeline Parallelism


# Why Profiling Matters

- Visualize computation, memory, communication.
- Identify bottlenecks early.
- Optimize GPU usage efficiently

# Single-GPU Training
<div class="column"  style="width:58%">
  ![](img/single_gpu.png){width=25%}
</div>
<div class="column"  style="width:40%">
  - <small>How it works: Entire model & data on one GPU.</small>
  - <small>Pros: Simple, fast for small models.</small>
  - <small>Cons: Not scalable to large models/datasets.</small>
</div>


# Data Parallelism (DP)
<div class="column"  style="width:58%">
  ![](img/data_parallelism.png){width=50%}
</div>
<div class="column"  style="width:40%">
  - <small>Copy model to each GPU.</small>
  - <small>Split inputs across GPUs.</small>
  - <small>Compute forward/backward.</small>
  - <small>Aggregate gradients.</small>
</div>    

# Naive Pytroch Data Parallelism (DP)
  ![](img/pytorch_dp_details.png){width=70%}


# Naive Pytroch Data Parallelism (DDP)
  ![](img/pytorch_ddp_details.png){width=70%}

 #Naive Pytroch Data Parallelism (DDP) (DDP)
- DP is Python threads-based, DDP is multiprocess-based 
  - No Python threads limitations, such as GIL
- Simpler data flow
- High inter-GPU communication overhead
- Overlapping pipeline of gradient all-reduce with layer gradient computation

# Tensor Parallelism
<div class="column"  style="width:58%">
  ![](img/tensor_parallelism.png){width=60%}
</div>
<div class="column"  style="width:40%">
  - <small>Send layers or blocks to different GPUs.</small>
  - <small>Transfer outputs between GPUs manually.</small>
</div>  

# Pipeline Parallelism
<div class="column"  style="width:50%">
  ![](img/pipeline_parallelism.png){width=60%}
</div>
<div class="column"  style="width:40%">
  <small>Idea: Split model layer-wise across GPUs.</small>
  <small>Each GPU processes part of the model sequentially, like a factory pipeline.</small>
  <small>Maximizes compute by overlapping stages (with microbatching).</small>
</div>

# Reality: 3D Parallelism
<div class="column"  style="width:100%; text-align: center;">
  ![](img/parallelism_3d.png){width=40%}
</div>
- In real world: Data Parallel + Tensor Parallel + Pipeline Parallel are combined.
- Example: Training GPT-3 used all three.


# ZeRO: Advance Data Parallelism
<div class="column"  style="width:100%; text-align: center;">
  ![](img/parallelism_zero.png){width=40%}
</div>
- Problem with Normal DP: Full optimizer states and gradients duplicated on every GPU.
- ZeRO Idea: Partition optimizer states, gradients, and parameters across GPUs.
- Result: Train MUCH larger models without running out of memory.


# Multi-GPU performance
<div class="column"  style="width:100%; text-align: center;">
  ![](img/GPU_overhead.png){width=40%}
</div>
- ResNet152 with CIFAR100 multi-gpu performance

# Demo: PyTorch Profiler
- Track CPU and GPU activities.
- Measure compute time, memory usage, communication overhead.
- Visualize using TensorBoard
- Demo
    1. Single-GPU Profiling
    2. Data Parallel Profiling
    3. Model Parallel Profiling
    4. TensorBoard Visualization

# Summary
- Model fits onto a single GPU -> DDP or ZeRO
- Model doesn’t fit onto a single GPU
  - Fast intra-node/GPU connection -> PP, ZeRO, TP
  - Without intra-node/GPU connection -> PP
- Largest Layer not fitting into a single GPU -> TP
- Multi-Node / Multi-GPU:
  - ZeRO - as it requires close to no modifications to the model
  - PP+TP+DP: less communications, but requires massive changes to the model
  - DP+PP+TP+ZeRO-1: when you have slow inter-node connectivity and still low on GPU memory
