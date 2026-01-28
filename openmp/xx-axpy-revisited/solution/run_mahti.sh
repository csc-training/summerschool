#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=test
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:10:00

g++ -std=c++20 -O3 -march=native -fopenmp axpy-malloc.cpp -o axpy.x

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_AFFINITY=true

run() {
    t=$1
    c=$2
    OMP_NUM_THREADS=$t srun --cpus-per-task=$c -o $(printf "axpy-t%03d-c%03d.out" "$t" "$c") ./axpy.x 102400000
}

run 1 1

for c in 2 4 8 16; do
    for t in 2 4 8 16; do
        if [ "$t" -le "$c" ]; then
            run $t $c
        fi
    done
done

for c in 16 32 64 128; do
    run $(( c / 8)) $c
done

for c in 16 32 64 128; do
    run $(( c / 16)) $c
done
