# Build

Build on the login node with `./build.sh`

# Run

Run with `sbatch profile.sbatch`

# Analyze

## Locally

1. `ssh` to LUMI
2. cd to this directory
3. run `. ../sourceme.sh`
4. run `omniperf analyze -p workloads/02_row_col/mi200/ --gui`
5. Wait until you see something similar to
```bash
--------
Analyze
--------

Dash is running on http://0.0.0.0:8050/

 * Serving Flask app 'omniperf_analyze.omniperf_analyze'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8050
 * Running on http://10.253.0.151:8050
Press CTRL+C to quit
```

6. Copy the public IP address (in the case above it's 10.253.0.151:8050)
7. Start a new terminal
8. On the new terminal run `ssh -L 8050:10.253.0.151:8050 username@lumi.csc.fi -N`
9. Open web browser
10. Go to `localhost:8050`

## Through LUMI web interface

1. Go to www.lumi.csc.fi
2. Start a desktop session 
3. Launch a terminal on the desktop session
4. cd to this directory
5. Do `. ../sourceme.sh`
6. run `omniperf analyze -p workloads/02_row_col/mi200/ --gui`
7. Open Firefox
8. Go to address `localhost:8050`
