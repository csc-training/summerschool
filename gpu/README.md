# GPU programming section

## Learning objectives

After completing the module, participants should be able to:


- Describe the key architectural features of modern GPUs and explain their implications for performance
- Explain the programming model of GPUs
- Develop and run GPU-accelerated applications using CUDA, HIP, and OpenMP offload
- Implement effective memory management strategies across host and accelerator environments
- Use performance analysis and profiling tools to identify and diagnose performance bottlenecks in GPU-enabled applications
- Implement applications utilizing multiple GPUs with MPI

## Demo codes

See [demos](demos/) for demo codes referred to in the slides.

## Sunday exercises

### Basics of HIP programming

- [Hello HIP API](exercises/02-kernels-hello-api)
- [Launching kernels](exercises/02-kernels-kernel-launch)
- [Kernel launch wrapper](exercises/02-kernels-kernel-launch-wrapper)
- [Using the API to query information](exercises/02-kernels-api-queries)
- [Catching API errors](exercises/02-kernels-api-errors)
- [Debugging kernels with prints](exercises/02-kernels-kernel-errors)
- [Fill kernel](exercises/02-kernels-fill)
- [Taylor for](exercises/02-kernels-taylor-for)
- [Copy2d](exercises/02-kernels-copy2d)

## Monday exercises

### Streams, events, and synchronization

- [Basics of streams](exercises/04-streams-basics)
- [Asynchronous kernels](exercises/04-streams-asynckernel)
- [Asynchronous memory copy](exercises/04-streams-asyncmemcopy)
- [Synchronizing with events](exercises/04-streams-eventssync)

### Memory management

- [API learning exercise](exercises/05-memory-basics)

### Kernel optimizations

- [Investigate coalescing](exercises/06-optimization-coalescing)
- [Tracing with rocprof](exercises/06-optimization-trace)

## Tuesday exercises

### OpenMP offloading

- [Hello world with OpenMP offloading](exercises/07-openmp-hello-world)
- [Using data environments](exercises/07-openmp-sum-dot)
- [Data movement clauses and reductions](exercises/07-openmp-dot-product)
- [Mimic HIP](exercises/07-openmp-mimic-hip)

### Multi-GPU programming

- [Ping-pong with multiple GPUs and MPI](exercises/09-multi-gpu-ping-pong)
- [Vector sum on two GPUs without MPI](exercises/09-multi-gpu-vector-sum)
- [Peer to peer device access](exercises/09-multi-gpu-p2pcopy)

## Bonus exercises

- [Basics of HIP: Debugging & fixing errors](exercises/02-kernels-kernel-errors)
- [Basics of HIP: Copy 2D kernel](exercises/02-kernels-copy2d)
- [Optimization: Matrix transpose](exercises/06-optimization-matrix-transpose)
- [OpenMP offloading: Heat equation](exercises/07-openmp-heat-equation)
- [OpenMP offloading: Interoperability with HIP](exercises/07-openmp-hip-interop)
- [OpenMP offloading: Jacobi](exercises/07-openmp-jacobi)


## Web resources

- [HIP documentation](https://rocm.docs.amd.com/en/latest/)
- [rocprof documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [CUDA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [OpenMP reference guides](https://www.openmp.org/resources/refguides/)
