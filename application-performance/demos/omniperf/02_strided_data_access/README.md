# Build

Build on the login node with `./build.sh`

# Run

Run with `sbatch profile.sbatch`

# Analyze

1. Go to www.lumi.csc.fi
2. Start a desktop session 
3. Launch a terminal on the desktop session
4. cd to this directory
5. Do `. ../sourceme.sh`
6. run `omniperf analyze -p workloads/01_three_kernels/mi200/ --gui`
7. Open Firefox
8. Go to address `localhost:8050`
9. Analyze
