# Demo: trace (and profile) with rocprof

Modules:
```bash
ml LUMI/24.03 partition/G PrgEnv-cray rocm
```

Compile
```bash
CC -xhip tracedemo.cpp -o tracedemo
```

## rocprof v1

Run and trace:
```bash
srun --job-name=example --account=project_462000956 --partition=small-g --reservation=SummerSchoolGPU --time=00:05:00 --gpus-per-node=1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 \
rocprof --hip-trace ./tracedemo
```

`results.json` appears, open it with [ui.perfetto.dev](ui.perfetto.dev) or [chrome://tracing](chrome://tracing).

## extra: rocprof v2

`rocprofv2` appears to fail to capture concurrent kernels but it can visualize performance metrics counters nicely.

Run:
```bash
srun --job-name=example --account=project_462000956 --partition=small-g --reservation=SummerSchoolGPU --time=00:05:00 --gpus-per-node=1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 \
rocprofv2 -i metrics.txt --plugin perfetto ./tracedemo
```
