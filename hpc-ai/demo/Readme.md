Work in progress...

# Demo

In this demo, we study image classification with [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. [This image](./img/cifar100.jpg) is a sample of what the dataset looks like. The dataset has 100 classes containing 600 images each.

We will train a CNN model called [ResNet152](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html). This model has over 60M parameters to train.

## Task 1

In the first task, we will use one single GPU to train the model Starting with `train_cifar100.py` and familiarize yourself with the codebase.

You can run the training directly with the corresponding script listed above:

    sbatch  run_single_cifar.sh

As a reminder, you can check the status of your runs with the command:

    squeue --me

The output of the run will appear in a file named `slurm-RUN_ID.out`
where `RUN_ID` is the Slurm batch job id. You can check the last ten
lines of that file with the command:

    tail slurm-RUN_ID.out

Use `tail -f` if you want to continuously follow the progress of the
output. (Press Ctrl-C when you want to stop following the file.)

## Task 2

Repeat the experiment with the `train_ddp_cifar100.py` Which trains the model with 2 GPUs with the [Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) technique.

You can run the training directly with the corresponding script listed above:

    sbatch  run_ddp_cifar100.sh


## GPU Utilization
You can monitor your GPU usage with the following:
```
srun --overlap --pty --jobid=<jobid> rocm-smi
srun --overlap --pty --jobid=<jobid> bash
```
Where `<jobid>` should be replaced. You can find the jobID for your job by looking at the queue:

    squeue --me

See [GPU-accelerated machine learning](https://docs.csc.fi/support/tutorials/gpu-ml/) documentations on LUMI for more information .

## Questions:
1. For each task, look at the GPU utilization, and VRAM. Discuss on how to increase the VRAM.
2. By looking at the `slurm-RUN_ID.out`, look for the training speed for the each iteration and also one epoch. Why the first epoch is slower than other epochs?
3. Why the total number of iterations are different in DDP?
4. Do you see any overheads for the DDP training?

   
## TensorBoard

You can use TensorBoard either via the LUMI web user interface.

### LUMI web interface

1. Log in via <https://www.lumi.csc.fi/>
2. Select menu item: Apps → TensorBoard
4. In the form:
   - Select course project: project_462000956
   - Specify the "TensorBoard log directory", it's where you have cloned the course repository plus "hpc-ai/logs", for example:
  `/scratch/project_462000956/$USER/summerschool/hpc-ai/demo`. You can run `pwd` in the terminal to find out the full path where you are working.
   - Leave rest at default settings
6. Click "Launch"
7. Wait until you see the "Connect to Tensorboard" button, then click that.
8. When you're done using TensorBoard, please go to "My Interactive Sessions" in the LUMI web user interface and "Cancel" the session. (It will automatically terminate once the reserved time is up, but it's always better to release the resource as soon as possible so that others can use it.)
