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
cpujobid=$(submit_job << 'EOF'
#!/bin/bash

#SBATCH --account=project_462000007
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=13G
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --exclusive

ml PrgEnv-cray

(srun CC -fopenmp -std=c++17 -O3 -Wall -Wextra -Wpedantic -pedantic-errors -o omp main.cpp) || { echo "Failed to build openMP code"; exit 1; }
(srun CC          -std=c++17 -O3 -Wall -Wextra -Wpedantic -pedantic-errors -o serial main.cpp) || { echo "Failed to build serial code"; exit 1; }

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=64

for i in {0..16..4} 
do
    srun ./serial $i > "serial_$i.dat"
    srun ./omp $i > "omp_$i.dat"
done

EOF
)

echo "Submitting gpu job"
gpujobid=$(submit_job << 'EOF'
#!/bin/bash

#SBATCH --account=project_462000007
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g

ml PrgEnv-cray
ml craype-accel-amd-gfx90a
ml rocm

(srun CC -xhip -std=c++17 -O3 -Wall -Wextra -Wpedantic -pedantic-errors -o hip main.cpp) || { echo "Failed to build hip code"; exit 1; }

for i in {0..16..4} 
do
    srun ./hip $i > "hip_$i.dat"
done
EOF
)

#echo "Submitting gnuplot job with dependency on jobs $cpujobid and $gpujobid"
#sbatch --dependency afterok:$cpujobid:$gpujobid << EOF
##!/bin/bash
#
##SBATCH --account=project_462000007
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
##SBATCH --time=00:01:00
##SBATCH --partition=debug
#
#echo "Loading modules"
#ml LUMI/24.03
#ml partition/C
#ml gnuplot/5.4.10-cpeGNU-24.03
#
#echo "Plotting problem size vs runtimes "
#gnuplot -e "\
#    set terminal png size 2000,2000 font \"default,50\"; \
#    set output \"runtimes.png\"; \
#    set style data linespoints; \
#    set key left top; \
#    set logscale x; \
#    set logscale y; \
#    set title \"Runtime of ax + y\"; \
#    set xlabel \"problem size\"; \
#    set ylabel \"time [ns]\"; \
#    set grid; \
#    set xrange [10:10000000000]; \
#    set yrange [10:10000000000]; \
#    set xtics rotate by 45 right; \
#    plot \"serial.dat\" title \"serial\"            lw 4.0 pt 5 ps 5, \
#         \"omp.dat\"    title \"OpenMP 64 threads\" lw 4.0 pt 7 ps 5, \
#         \"hip.dat\"    title \"gpu\"               lw 4.0 pt 9 ps 5; \
#    "
#EOF
