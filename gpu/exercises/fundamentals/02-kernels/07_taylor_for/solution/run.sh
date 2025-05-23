##!/bin/bash
#
##SBATCH --job-name=taylor_measurement
##SBATCH --account=project_462000007
##SBATCH --partition=standard-g
##SBATCH --time=00:10:00
##SBATCH --tasks=1
##SBATCH --nodes=1
##SBATCH --gpus-per-node=1
##SBATCH --cpus-per-task=1
##SBATCH --mem=0
##SBATCH --exclusive
#
#ml PrgEnv-amd
#
#export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
#export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)
#
#hipcc -O3 main.cpp

echo "nti, vs, us, vec, strided, consecutive, vec_for" > runtimes.dat

for i in {0..5}
do
    num_taylor_iters=$((1 << $i))
    for j in {0..6}
    do
        vec_size=$(((1 << $j) * 1000000))
        echo -n "$num_taylor_iters, $vec_size, " >> runtimes.dat
        srun ./a.out $num_taylor_iters $vec_size >> runtimes.dat
    done
done
