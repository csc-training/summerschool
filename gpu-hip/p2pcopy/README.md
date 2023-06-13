# Peer to peer device access

Benchmark memory copies with and without peer to peer device access with HIP using two GPUs, and try doing the same with OpenMP.

NOTE: Remember to request 2 GPUs when running this exercise. On Lumi, use
```
srun --account=XXXXXX --partition=dev-g -N1 -n1 --cpus-per-task=1 --gpus-per-task=2 --time=00:15:00 ./a.out
```
and on Puhti use
```
srun --account=XXXXXX --partition=gputest -N1 -n1 --cpus-per-task=1 --gres=gpu:v100:2 --time=00:15:00 ./a.out
```

## Case 1 - HIP
Skeleton code [hip-p2pcopy.cpp](hip-p2pcopy.cpp) tests peer to peer device access between two GPUs by doing a series of memory copies. The test is evaluated after calling `hipDeviceEnablePeerAccess()` and `hipDeviceDisablePeerAccess()`. The program prints calculated bandwith and time for both cases. On a CUDA platform, there should be a difference in results, whereas on an AMD platform there is none. In order to make the code work, you need to fix the missing parts marked with
TODOs.

## Case 2 - OpenMP
First, some modules are needed.

On **Lumi**, load the PrgEnv-amd instead of PrgEnv-cray as follows:
```
module purge
module load PrgEnv-amd
module load craype-accel-amd-gfx90a
module load rocm
```
This is because Cray compiler has no multi-gpu support for `omp_target_alloc()`.

Copy [hip-p2pcopy.cpp](hip-p2pcopy.cpp) into [omp-p2pcopy.cpp](omp-p2pcopy.cpp) and modify the code to use OpenMP instead of HIP. With OpenMP, you can't check, enable or disable peer access like with HIP, so these parts of the code can be removed. You may find `omp_target_alloc()` and `omp_target_free()` and `omp_target_memcpy()` functions useful for device memory management with OpenMP. Does it look like the direct peer to peer access works properly with OpenMP, when comparing the bandwith between Case 1 and Case 2?

