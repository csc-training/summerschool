---
title:  "Scaling PyTorch Models: Single vs Multi-GPU Training and Techniques"
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Agenda

- Single-GPU Profiling
- Data Parallelism
    - Basic Data Parallel (DP)
    - ZeRO Data Parallel (DeepSpeed)
- Model Parallelism
    - Tensor Parallelism
    - Pipeline Parallelism
- Profiling and Analysis


# Why Profiling Matters

- Visualize computation, memory, communication.
- Identify bottlenecks early.
- Optimize GPU usage efficiently


# Model Parameters and GPU FLops

- Floating point operations per second
- Simple example


# Signle-GPU Training
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
  - <small>1. Copy model to each GPU.</small>
  - <small>2. Split inputs across GPUs.</small>
  - <small>3. Compute forward/backward independently.</small>
  - <small>4. Aggregate gradients (AllReduce).</small>
</div>    


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
  - <small>Idea: Split model layer-wise across GPUs.</small>
  - <small>Each GPU processes part of the model sequentially, like a factory pipeline.</small>
  - <small>Maximizes compute by overlapping stages (with microbatching).</small>
</div>

# Realliity: 3D Parallelism
<div class="column"  style="width:100%; text-align: center;">
  ![](img/parallelism_3d.png){width=40%}
</div>
- In real world: Data Parallel + Tensor Parallel + Pipeline Parallel are combined.
- Example: Training GPT-3 used all three.


# Advance Data Parallelism - ZeRO and DDP
<div class="column"  style="width:100%; text-align: center;">
  ![](img/parallelism_zero.png){width=40%}
</div>
- Problem with Normal DP: Full optimizer states and gradients duplicated on every GPU.
- ZeRO Idea: Partition optimizer states, gradients, and parameters across GPUs.
- Result: Train MUCH larger models without running out of memory.


# Demo: PyTorch Profiler
- Track CPU and GPU activities.
- Measure compute time, memory usage, communication overhead.
- Visualize using TensorBoard
- Demo
    1. Single-GPU Profiling
    2. Data Parallel Profiling
    3. Model Parallel Profiling
    4. TensorBoard Visualization


# Excerices
- How much is the communication overhead?
- How to reduce communication overhead?
- How to better split model for model parallelism?
- Can we balance GPU loads better?