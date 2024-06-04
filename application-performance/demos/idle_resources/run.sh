#!/bin/bash

submit_job() {
  sub="$(sbatch "$@")"

  if [[ "$sub" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    exit 1
  fi
}

echo "Submitting cpu job"
cpujobid=$(submit_job << "EOF"
#!/bin/bash

#SBATCH --account=project_462000007
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:05:00
#SBATCH --partition=small
#SBATCH --exclusive

ml PrgEnv-cray

(srun CC -O3 -fopenmp -o omp omp_saxpy.cpp) || { echo "Failed to build openMP code"; exit 1; }
(srun CC -O3 -o serial serial_saxpy.cpp) || { echo "Failed to build serial code"; exit 1; }

export OMP_PROC_BIND=close
export OMP_PLACES=cores

for nthreads in 2 64
do
    OMP_NUM_THREADS=$nthreads srun ./omp > "omp$nthreads.dat"
done

srun ./serial > "serial.dat"
EOF
)

echo "Submitting gpu job"
gpujobid=$(submit_job << EOF
#!/bin/bash

#SBATCH --account=project_462000007
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=dev-g

ml PrgEnv-cray
ml craype-accel-amd-gfx90a
ml rocm

(srun CC -std=c++17 -xhip -O3 -o hip hip_saxpy.cpp) || { echo "Failed to build hip code"; exit 1; }
srun ./hip > "hip.dat"
EOF
)

echo "Submitting gnuplot job with dependency on jobs $cpujobid and $gpujobid"
sbatch --dependency afterok:$cpujobid:$gpujobid << EOF
#!/bin/bash

#SBATCH --account=project_462000007
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=small

echo "Loading modules"
ml LUMI/23.09
ml partition/L
ml gnuplot/5.4.8-cpeGNU-23.09

echo "Plotting problem size vs runtimes "
gnuplot -e "\
    set terminal png size 1000,1000; \
    set output \"runtimes.png\"; \
    set style data linespoints; \
    set key left top; \
    set logscale x; \
    set logscale y; \
    set title \"Runtime of Ax + y with different implementation strategies\"; \
    set xlabel \"problem size\"; \
    set ylabel \"time [ns]\"; \
    set grid; \
    set xrange [10:1000000000]; \
    plot \"serial.dat\" title \"serial\" lw 2.5, \
        \"omp2.dat\" title \"OpenMP 2 threads\" lw 2.5, \
        \"omp64.dat\" title \"OpenMP 64 threads\" lw 2.5, \
        \"hip.dat\" title \"gpu\" lw 2.5; \
    " 
EOF
