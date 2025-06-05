## Prerequisite

Load modules to use EasyBuild, then build Score-P and Scalasca
(building Score-P/Scalasca may take tens of minutes):

```bash
ml LUMI/24.03
ml partition/C
ml EasyBuild-user/LUMI

eb Score-P-9.0-cpeCray-24.03.eb -r
eb Scalasca-2.6.2-cpeCray-24.03.eb -r
```

Clone the heat equation code.

```bash
git clone https://github.com/cschpc/heat-equation.git
cd heat-equation/2d/mpi-openmp
```

## Demo

First build normally:

```bash
make CXX="CC"\
     CC="cc"\
     FC="ftn"
```

Run the program on LUMI to get a reference runtime:

```bash
OMP_NUM_THREADS=16\
srun\
    -A project_462000007\
    -N 4\
    -n 16\
    -c 16\
    -t 00:10:00\
    -p standard\
./heat_hybrid 8000 8000 20000
```

Then build with instrumentation:

```bash
make clean
make CXX="scorep CC"\
     CC="scorep cc"\
     FC="scorep ftn"
```

Then run an initial summary measurement to figure out the overhead of the instrumentation:

```bash
OMP_NUM_THREADS=16\
scalasca\
    -analyze\
srun\
    -A project_462000007\
    -N 4\
    -n 16\
    -c 16\
    -t 00:10:00\
    -p standard\
./heat_hybrid 8000 8000 20000
```

Score the summary:

```bash
scalasca -examine -s scorep_heat_hybrid
```
