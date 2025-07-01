#!/bin/bash
#SBATCH --account=project_462000956
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=0:30:00
#SBATCH --output=run_model_parameters-%j.out

module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

SIF=/scratch/project_462000956/resources/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

srun singularity exec $SIF bash -c \
    '$WITH_CONDA; \
    source /scratch/project_462000956/resources/hpc-ai/bin/activate; \
    python get_parameters.py'