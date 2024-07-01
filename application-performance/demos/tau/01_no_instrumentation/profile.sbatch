#!/bin/bash -l

#SBATCH --account=project_465001194
#SBATCH --job-name=hip_heat
#SBATCH --output=hip_heat.out%j
#SBATCH --error=hip_heat.err%j
#SBATCH --partition=small-g
#SBATCH --reservation=CSC_summer_school_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --mem=10G
#SBATCH --time=00:05:00

ml LUMI/23.09
ml partition/G
ml PrgEnv-cray
ml craype-accel-amd-gfx90a
ml rocm/5.4.6

export MPICH_GPU_SUPPORT_ENABLED=1

. ../sourceme.sh

export TAU_PROFILE=1
export PROFILEDIR=tauprofile
[ -d $PROFILEDIR ] || mkdir $PROFILEDIR

srun tau_exec \
    -T mpi,rocprofiler \
    -ebs \
    src/hip/heat_hip 4000 4000 5000
