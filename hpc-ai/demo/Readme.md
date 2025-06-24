# Demos
In this demo, we will demonstrate some of the basic techniques in the second lecture. We will demonstrate how to use Tensorboard along with Pytorch Profiler to track the GPU usage. In these examples, we are using `ResNet152`.

## How to run the demo
There are three files in the demo folder: `train_single_gpu.py`, `train_data_parallel.py` and `train_ddp.py`.

## Tensorboard
To visualize the GPU traces, we are using Tensorboard in this example. Tensorboard is the standard approach in the AI community. More recently, using Weights & Biases website is more common, but we are not covering that in this lecture.

To use Tensorboard in LUMI, follow this steps:
1. Go to (Lumi)[https://www.lumi.csc.fi/public/]
2. Login using your credtials
3. In the page, click on Tensorboard.
4. In the form, under the Setting -> TensorBoard log directory, put the directory with the log information for the profiler.
