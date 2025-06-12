## Prerequisite

**TODO: Build these before hand**

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

**TODO: Try to get hybrid working**
**TODO: Tracing requires pragma omp master & pragma omp barrier**

```bash
git clone https://github.com/cschpc/heat-equation.git
cd heat-equation/2d/mpi
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
srun \
    --account=project_46200095 \
    -N 2 \
    -n 128 \
    -c 1 \
    -t 00:10:00 \
    -p standard \
./heat_mpi 4096 4096 20000
```

Then build with instrumentation:

```bash
ml Score-P/9.0-cpeCray-24.03

make clean
make CXX="scorep CC"\
     CC="scorep cc"\
     FC="scorep ftn"
```

Then run an initial summary measurement to figure out the overhead of the instrumentation:

```bash
ml Scalasca/2.6.2-cpeCray-24.03

scalasca \
    -analyze \
srun \
    --account=project_46200095 \
    -N 2 \
    -n 128 \
    -c 1 \
    -t 00:10:00 \
    -p standard \
./heat_mpi 4096 4096 20000
```

Score the summary and view 25 rows:

```bash
scalasca -examine -s scorep_heat_mpi_2p128_sum
head -n 25 scorep_heat_mpi_2p128_sum/scorep.score
```

Generate a filter:

```bash
scorep-score -m -g scorep_heat_mpi_2p128_sum/profile.cubex
```

Score it with the filter:

```bash
scalasca -examine -s -f initial_scorep.filter scorep_heat_mpi_2p128_sum/
```

Rename the unfiltered score, otherwise Scalasca will abort immediately,
as the directory exists:

```bash
mv scorep_heat_mpi_2p128_sum/ scorep_heat_mpi_2p128_sum-unfiltered/
```

Run with the filter:

```bash
scalasca \
    -analyze \
    -f initial_scorep.filter \
srun \
    --account=project_46200095\
    -N 2 \
    -n 128 \
    -c 1 \
    -t 00:10:00 \
    -p standard \
./heat_mpi 4096 4096 20000

scalasca -examine -s scorep_heat_mpi_2p128_sum/
```

Download CubeGUI to your own laptop for viewing the profile,
download the summary from LUMI and open it with CubeGUI:

```bash
wget https://apps.fz-juelich.de/scalasca/releases/cube/4.9/dist/CubeGUI-4.9.AppImage
chmod +x CubeGUI-4.9.AppImage
scp lumi:/users/juhanala/Documents/summerschool/application-performance/demos/scalasca/heat-equation/2d/mpi/scorep_heat_mpi_2p128_sum/summary.cubex .
./CubeGUI-4.9.AppImage summary.cubex
```

Rerun with fewer ranks:

```bash
scalasca \
    -analyze \
srun \
    --account=project_46200095 \
    -N 1 \
    -n 64 \
    -c 1 \
    -t 00:10:00 \
    -p standard \
./heat_mpi 4096 4096 20000

scalasca -examine -s scorep_heat_mpi_1p64_sum/
```

Tracing:

```bash
OMP_NUM_THREADS=16 \
SCOREP_TOTAL_MEMORY=48MB \
scalasca \
    -analyze \
    -f initial_scorep.filter \
    -q \
    -t \
srun \
    --account=project_46200095 \
    -N 2 \
    -n 16 \
    -c 16 \
    -t 00:10:00 \
    -p standard \
./heat_hybrid 4096 4096 5000
```

## On Mahti

```bash
ml gcc/11.2.0
ml openmpi/4.1.2
ml scorep/7.0
ml scalasca/2.6

OMP_NUM_THREADS=16 \
SCOREP_TOTAL_MEMORY=48MB \
scalasca \
    -analyze \
    -f initial_scorep.filter \
    -q \
    -t \
srun \
    --account=project_46200095 \
    -N 2 \
    -n 16 \
    -c 16 \
    -t 00:10:00 \
    -p test \
./heat_hybrid 4096 4096 5000
```
