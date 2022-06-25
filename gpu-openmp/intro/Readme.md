# Device query on Puhti

## 1) Start an interactive session with 1 GPU:

```
salloc --partition=gpu --time=00:15:00 --ntasks=1 --cpus-per-task=10 \
  --mem-per-cpu=8000 --gres=gpu:v100:1 --job-name=d_query \
  --account=<training_project> --reservation=<reservation name>
```

Log in on the running node `ssh rxxgxx`.

1. Load the hip module, then run the command `nvidia-smi`. Investigate the
   output
2. Run the command `$CUDA_INSTALL_ROOT/extras/demo_suite/deviceQuery`. Comment
   on the output
3. Check more examples from `$CUDA_INSTALL_ROOT/extras/demo_suite/`.


### 1B) Alternatively, use a non-interactive job

Instead of using an interactive session, run all the commands above in a
non-interactive way by adding the commands to the submission script below.

```
#!/bin/bash
#SBATCH --job-name=d_query
#SBATCH --account=<training_project>
#SBATCH --reservation=<reservation name>
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1

srun nvidia-smi
```

## 2) Repeat the tasks above with 2 or more GPUs.
