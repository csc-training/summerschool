#!/bin/bash

#SBATCH --job-name=taylor_measurement
#SBATCH --account=project_462000956
#SBATCH --partition=standard-g
#SBATCH --time=00:20:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive

ml PrgEnv-amd

export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)

hipcc -O3 main.cpp

echo "Taylor N, bytes, block size, Base, Vectorized, Strided, Consecutive, Strided vectorized" > runtimes.dat

for i in {0..3}
do
    # 1, 2, 4, 8
    num_taylor_iters=$((1 << $i))
    for j in {0..6}
    do
        # 1e6, 2e6, 4e6, 8e6, 16e6, 32e6, 64e6
        vec_size=$(((1 << $j) * 1024 * 1024))
        for k in {0..12}
        do
            # 256, 320, 384, ..., 1024
            block_size=$((256 + $k * 64))
            srun ./a.out $num_taylor_iters $vec_size $block_size >> runtimes.dat

            if [ $? -ne 0 ]
            then
                echo "Error with srunning application"
                exit 1
            fi
        done
    done
done
