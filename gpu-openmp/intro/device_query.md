# Device query on Puhti

A) Start an interactive session with 1 GPU:

```
salloc --job-name=d_query --account=<training_project> --partition=gpu --reservation=<rezervation name> --time=00:15:00 --ntasks=1  --cpus-per-task=10 --mem-per-cpu=8000 --gres=gpu:v100:1
``` 

Log in on the running node `ssh rxxgxx`.

1. Load the hip module, then run the command `nvidia-smi`. Investigate the output
2. Run the command `$CUDA_INSTALL_ROOT/extras/demo_suite/deviceQuery`. Comment on the output

B) Start an interactive session with 2 or more GPUs and repeat the tasks above. 

C) With 1 or more GPUs, check more examples from `$CUDA_INSTALL_ROOT/extras/demo_suite/`. 


Alternatevely run all these commands in non-interactive way by using the submission script:

```
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --account=<training_project>
#SBATCH --reservation=<rezervation name>
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10 
#SBATCH --mem-per-cpu=8000 
#SBATCH --gres=gpu:v100:1

srun nvidia-smi
```
