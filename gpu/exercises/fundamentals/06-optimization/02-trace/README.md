# Tracing with rocprof

In this exercise your task is to trace execution of [streams/02-concurrency](../../streams/02-concurrency/solution/streams.cpp) exercise
solution.

Rocprof can be used to trace HIP API calls, among others, with option

```bash
> rocprof --hip-trace <executable>
```

It will output a file named `results.json` which may be visualized for example
with perfetto trace visualizer (https://ui.perfetto.dev/) or chrome/chromium
built in visualizer tools (type `chrome://tracing/` in the URL field).

## Exercise

- Trace the HIP API calls of the `streams.cpp` code and visualize the results.
- Modify `WORK` preprocessor macro to so large that kernel executions begin to
  exceed memory transfers.
- Does the kernel execution order correspond to their stream numbering?
