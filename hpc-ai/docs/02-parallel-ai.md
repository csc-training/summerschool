---
title:  "Scaling PyTorch Models: Single vs Multi-GPU Training and Techniques"
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Single GPU vs Multi-GPU training
- Training ML models could be intense
  - Heavy computations
  - Large Model
- That's why we might need use multiple GPUs to train
  - GPUs could be accross multiple nodes
- Multi-GPU or Multi-Node training has a over head
  - Communication costs
  - Distributation of the data

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
  ![](img/pytorch_dp_details.png){width=75%}


# Pytroch Distributed Data Parallelism (DDP)
  ![](img/pytorch_ddp_details.png){width=75%}


# DDP vs DP
- DP is Python threads-based, DDP is multiprocess-based 
  - No Python threads limitations, such as GIL
- Simpler data flow
- DDP has a High inter-GPU communication overhead (all-reduce)
- Overlapping pipeline of gradient all-reduce with layer gradient computation


# Pipeline Parallelism
<div class="column"  style="width:50%">
  ![](img/pipeline_parallelism.png){width=60%}
</div>
<div class="column"  style="width:40%">
  <small>Idea: Split model layer-wise across GPUs.</small>
  <small>Each GPU processes part of the model sequentially, like a factory pipeline.</small>
  <small>Maximizes compute by overlapping stages (with microbatching).</small>
</div>


# Bubble issue and GPipe
<div class="column"  style="width:80%; text-align: center;">
  ![](img/mp_vs_gpipe.png){width=80%}
  - <small>Picture from [GPipe Paper, arXiv:1811.06965](https://arxiv.org/abs/1811.06965)</small>
</div>
- GPipe divided the data to micro-batch to reduce the bubble issue.


# Tensor Parallelism
<div class="column"  style="width:58%">
  ![](img/tensor_parallelism.png){width=60%}
</div>
<div class="column"  style="width:40%">
  - <small>Send layers or blocks to different GPUs.</small>
  - <small>Transfer outputs between GPUs manually.</small>
</div>  


# How TP works?
<div class="column"  style="width:80%; text-align: center;">
  ![](img/tp_example.png){width=80%}
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
