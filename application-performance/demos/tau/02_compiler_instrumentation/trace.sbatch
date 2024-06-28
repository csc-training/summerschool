#!/bin/bash -l

#SBATCH --account=project_465001194
#SBATCH --job-name=hip_heat
#SBATCH --output=hip_heat.out%j
#SBATCH --error=hip_heat.err%j
#SBATCH --partition=small-g
#SBATCH --reservation=CSC_summer_school_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=10G
#SBATCH --time=00:05:00

ml LUMI/23.09
ml partition/G
ml PrgEnv-cray
ml craype-accel-amd-gfx90a
ml rocm/5.4.6

cat << "EOF" > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*
EOF

chmod +x ./select_gpu
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
export MPICH_GPU_SUPPORT_ENABLED=1

. ../sourceme.sh

export TAU_TRACE=1
export TRACEDIR=tautrace
[ -d $TRACEDIR ] || mkdir $TRACEDIR

srun --cpu-bind=${CPU_BIND} \
    ./select_gpu \
    src/hip/heat_hip 10000 10000 5000

rm -rf ./select_gpu
