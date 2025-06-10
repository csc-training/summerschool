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

## Case 2 - OpenMP Offloading
First, some changes in modules are needed.

On **LUMI**, swap the loaded PrgEnv-cray to PrgEnv-cray-amd as follows:
```
module load PrgEnv-cray-amd
```
This is because Cray compiler has no multi-gpu support for `omp_target_alloc()`.

Copy [hip-p2pcopy.cpp](hip-p2pcopy.cpp) into [omp-p2pcopy.cpp](omp-p2pcopy.cpp) and modify the code to use OpenMP instead of HIP. With OpenMP, one can't check, enable or disable peer access like with HIP, so these parts of the code can be removed. You may find `omp_target_alloc()` and `omp_target_free()` and `omp_target_memcpy()` functions useful for device memory management with OpenMP. Does it look like the direct peer to peer access works properly with OpenMP, when comparing the bandwith between Case 1 and Case 2?

## Case 3 - SYCL  & USM
With SYCL as well, one can't check, enable or disable peer access like with HIP, so these parts of the code can be removed. While the pointers allocated with `usm_device()` are associated with a specific queue, if the devices are in the same platform one can use the `memcpy()` method, `q0.memcpy(dA_0, dA_1, sizeof(int)*N);`. Under the hood it should use the peer to peer transfers.  Does it look like the direct peer to peer access works properly with SYCL, when comparing the bandwith between Case 1 and Case 2?

**Note!** In both OpenMP offloadingg and SYCL, if needed, one can mix the code  HIP/CUDA API. This however breaks the portability idea behind the two parallel proggrgaming models. 