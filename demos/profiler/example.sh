## Example commands to try
##   note: for NVVP, you need to log in with 'ssh -X ...'

# NVPROF
srun -N1 -n1 -pgpu --gres=gpu:p100:1 --reservation=Summerschool nvprof ./vectorsum
srun -N1 -n1 -pgpu --gres=gpu:p100:1 --reservation=Summerschool nvprof ./heat_openacc

# NVVP
srun -N1 -n1 -pgpu --gres=gpu:p100:1 --reservation=Summerschool --x11=first --mem=52000 nvvp ./vectorsum
srun -N1 -n1 -pgpu --gres=gpu:p100:1 --reservation=Summerschool --x11=first --mem=52000 nvvp ./heat_openacc
