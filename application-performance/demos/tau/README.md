# Build on LUMI

1. run `. sourceme.sh`
2. cd to one of the subdirectories
3. Use `./build.sh` to build the code on a login node
4. `sbatch profile.sbatch` to profile the code or `sbatch trace.sbatch` to trace the code

# Analyze
If you **profiled** the code you can either
1. cd `tauprofile`
2. run `pprof` to use the cli interface

or
1. go to `www.lumi.csc.fi` on your browser
2. launch a desktop session
3. open terminal on the session
4. cd to the `tauprofile` dir
5. run `. ../../sourceme.sh`
6. run `paraprof`

If you **traced** the code do
1. `cd tautrace`
2. `tau_treemerge.pl`
3. `tau_trace2json tau.trc tau.edf -chrome -ignoreatomic -o app.json`
4. copy the app.json to your laptop (or use lumi.csc.fi)
5. go to `https://ui.perfetto.dev`
6. Open trace file `app.json`
