# Device query on Puhti

A) Start an interactive session with 1 GPU:

```
salloc --job-name=d_query --account=<training_project> --partition=gpu --rezervation=<rezervation name> --time=01:00:00 --ntasks=1  --cpus-per-task=10 --mem-per-cpu=8000 --gres=gpu:v100:1
``` 

Log in on the running node `ssh rxxgxx`.

1. Load the hip module, then run the command `nvidia-smi`. Investigate the output
2. Run the command `$CUDA_INSTALL_ROOT/extras/demo_suite/deviceQuery`. Comment on the output

B) Start an interactive session with 2 or more GPUs and repeat the tasks above. 
