#!/bin/bash
#SBATCH --job-name=data_parallel_cifar100
#SBATCH --account=project_462000956
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=60G
#SBATCH --time=0:30:00
#SBATCH --output=data_parallel_cifar100-%j.out
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

SIF=/scratch/project_462000956/resources/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

srun singularity exec $SIF bash -c \
    '$WITH_CONDA; \
    source /scratch/project_462000956/resources/hpc-ai/bin/activate; \
    python train_data_parallel_cifar100.py'