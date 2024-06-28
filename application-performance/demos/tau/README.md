# Build

1. run `. sourceme.sh`
2. cd to one of the subdirectories
3. Use `./build.sh` to build the code on a login node
4. `sbatch profile.sbatch` to profile the code or `sbatch trace.sbatch` to trace the code

If you traced the code do
1. `cd tautrace`
2. `tau_treemerge.pl`
3. `tau_trace2json tau.trc tau.edf -chrome -ignoreatomic -o app.json`
4. copy the app.json to your laptop (or use lumi.csc.fi)
5. go to `https://ui.perfetto.dev`
6. Open trace file `app.json`
