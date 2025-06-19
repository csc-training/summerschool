## Prerequisite

Clone the heat equation code.

```bash
mkdir -p /scratch/project_462000956/$USER
cd /scratch/project_462000956/$USER
git clone https://github.com/cschpc/heat-equation.git
cd heat-equation/2d/mpi-openmp
```

## Demo

First build normally:

```bash
bash << 'EOF'
make CXX="CC"\
     CC="cc"\
     FC="ftn"
EOF
```

Run the program on LUMI to get a reference runtime:

```bash
sbatch << 'EOF'
#!/bin/bash

#SBATCH -A project_462000956
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -c 16
#SBATCH -t 00:10:00
#SBATCH -p standard

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun ./heat_hybrid 4096 4096 20000
EOF
```

Then build with instrumentation:

```bash
bash << 'EOF'
#!/bin/bash

export EBU_USER_PREFIX=/projappl/project_462000956/EB/

ml LUMI/24.03
ml partition/C
ml Score-P/9.0-cpeGNU-24.03

rm ../../common/*.o
make clean
make CXX="scorep CC"\
     CC="scorep cc"\
     FC="scorep ftn"
EOF
```

Then run an initial summary measurement to figure out the overhead of the instrumentation:

```bash
sbatch << 'EOF'
#!/bin/bash

#SBATCH -A project_462000956
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -c 16
#SBATCH -t 00:10:00
#SBATCH -p standard

export EBU_USER_PREFIX=/projappl/project_462000956/EB/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SCOREP_EXPERIMENT_DIRECTORY=scorep_experiment_${SLURM_JOBID}

ml LUMI/24.03
ml partition/C
ml Score-P/9.0-cpeGNU-24.03
ml Scalasca/2.6.2-cpeGNU-24.03

# Run the analysis
scalasca \
    -analyze \
srun ./heat_hybrid 4096 4096 20000

# Score the summary
scalasca -examine -s ${SCOREP_EXPERIMENT_DIRECTORY}

# Print out 40 first lines of the score
head -n 40 ${SCOREP_EXPERIMENT_DIRECTORY}/scorep.score

# Generate an initial filter
scorep-score -m -g ${SCOREP_EXPERIMENT_DIRECTORY}/profile.cubex

# Rescore with the filter
scalasca -examine -s -f initial_scorep.filter ${SCOREP_EXPERIMENT_DIRECTORY}/

head -n 40 ${SCOREP_EXPERIMENT_DIRECTORY}/scorep.score
EOF
```

Run with the filter:

```bash
sbatch << 'EOF'
#!/bin/bash

#SBATCH -A project_462000956
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -c 16
#SBATCH -t 00:10:00
#SBATCH -p standard

export EBU_USER_PREFIX=/projappl/project_462000956/EB/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SCOREP_EXPERIMENT_DIRECTORY=scorep_experiment_${SLURM_JOBID}

ml LUMI/24.03
ml partition/C
ml Score-P/9.0-cpeGNU-24.03
ml Scalasca/2.6.2-cpeGNU-24.03

# Run the analysis
scalasca \
    -analyze \
    -f initial_scorep.filter \
srun ./heat_hybrid 4096 4096 20000

scalasca -examine -s ${SCOREP_EXPERIMENT_DIRECTORY}
head -n 40 ${SCOREP_EXPERIMENT_DIRECTORY}/scorep.score
EOF
```

Download CubeGUI to your own laptop for viewing the profile,
download the summary from LUMI and open it with CubeGUI:

```bash
wget https://apps.fz-juelich.de/scalasca/releases/cube/4.9/dist/CubeGUI-4.9.AppImage
chmod +x CubeGUI-4.9.AppImage
scp lumi:/scratch/project_462000956/juhanala/heat-equation/2d/mpi-openmp/scorep_experiment_11503172/trace.cubex .
./CubeGUI-4.9.AppImage summary.cubex
```

## Tracing

```bash
sbatch << 'EOF'
#!/bin/bash

#SBATCH -A project_462000956
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -c 16
#SBATCH -t 00:10:00
#SBATCH -p standard

export EBU_USER_PREFIX=/projappl/project_462000956/EB/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SCOREP_EXPERIMENT_DIRECTORY=scorep_experiment_${SLURM_JOBID}
export SCOREP_TOTAL_MEMORY=88MB

ml LUMI/24.03
ml partition/C
ml Score-P/9.0-cpeGNU-24.03
ml Scalasca/2.6.2-cpeGNU-24.03

export SCAN_TRACE_FILESYS=${PWD}${SCAN_TRACE_FILESYS:+:${SCAN_TRACE_FILESYS}}

scalasca \
    -analyze \
    -f initial_scorep.filter \
    -q \
    -t \
srun ./heat_hybrid 4096 4096 20000

scalasca -examine -s ${SCOREP_EXPERIMENT_DIRECTORY}
EOF
```

TODO: view with Vampir on LUMI
