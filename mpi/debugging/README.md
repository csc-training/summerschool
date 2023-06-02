# Debugging

## General notes

- Code needs to be compiled with debugging option `-g`
- Compiler optimizations might complicate debugging (dead code
  elimination, loop transformations, *etc.*), recommended to
  compile without optimizations with `-O0`
    - Sometimes bugs show up only with optimizations


## Launching Arm DDT

### LUMI

Set up VNC for smoother GUI performance:
```bash
module load lumi-vnc
start-vnc
```
Follow the instructions from the command output to set up
SSH port forwarding and opening VNC on a browser.

Load module and start debugger in an interactive session:
```bash
module load ARMForge
export SLURM_OVERLAP=1
salloc -A project_465000536 --nodes=1 --ntasks-per-node=2 --time=00:30:00 --partition=debug
ddt srun ./buggy
```

The debugger GUI will open in VNC in the browser window.

Note: you can also skip the VNC step and use X11 forwarding for GUI (might be slow).


### Mahti

Set up VNC for smoother GUI performance:
```bash
sinteractive -A project_2007995 --cores 1
module load vnc
start-vnc
```
Follow the instructions from the command output to set up
SSH port forwarding and opening VNC on a browser.

Load module and start debugger in an interactive session:
```bash
module load ddt
export SLURM_OVERLAP=1
salloc -A project_2007995 --nodes=1 --ntasks-per-node=2 --time=00:15:00 --partition=test
ddt srun ./buggy
```

The debugger GUI will open in VNC in the browser window.

Note: you can also skip the VNC step and use X11 forwarding for GUI (might be slow).


### Puhti

Launch a desktop session on a browser for smoother GUI performance:
* Login to https://www.puhti.csc.fi
* Launch Desktop with 1 core
* Open terminal in the desktop session and proceed there

Load module and start debugger in an interactive session:
```bash
module load ddt
export SLURM_OVERLAP=1
salloc -A project_2007995 --nodes=1 --ntasks-per-node=2 --time=00:15:00 --partition=test
ddt srun ./buggy
```

Note: you can also skip the desktop session step and use X11 forwarding for GUI (might be slow).

