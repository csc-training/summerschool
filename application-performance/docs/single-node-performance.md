# Single node performance analysis {.section}

# Performance analysis

![](images/perf-analysis-single-core.svg){.center width=60%}

# Introduction
- Don’t speculate about performance – measure it!
- Performance analysis tools help to
    - Find hot-spots
    - Identify the cause of less-than-ideal performance
- Tools covered here
    - Intel VTune Amplifier
- Other tools
    - Perf, gprof, CrayPAT, Tau, Scalasca, PAPI…
    - <http://www.vi-hps.org/tools/tools.html>

# Profiling application

- Collecting all possible performance metrics with single run is not practical
    - Simply too much information
    - Profiling overhead can alter application behavior
- Start with an overview!
    - Call tree information, what routines are most expensive?

# Sampling vs. tracing

- When application is profiled using sampling, the execution is stopped at 
  predetermined intervals and the state of the application is examined
    - Lightweight, but may give skewed results
- Tracing records every event, e.g. function call
    - Usually requires modification to the executable
        - These modifications are called instrumentation
    - More accurate, but may affect program behavior
    - Generates lots of data

# Hardware performance counters

- Hardware performance counters are special registers on CPU that count 
  hardware events
- They enable more accurate statistics and low overhead
    - In some cases they can be used for tracing without any extra 
      instrumentation

- Number of counters is much smaller than the number of events that can be 
  recorded
- Different CPUs have different counters

# Intel VTune 

- VTune is a tool that can give detailed information on application resource 
  utilization
    - Uses CPU hardware counters on Intel CPUs for more accurate statistics
- VTune has extensive GUI for result analysis and visualization

# VTune

- Analysis in three steps
    1. **Collect:** Run binary and collect performance data – sampling based 
       analysis
    2. **Finalize:** Prepare data for analysis – by default combined with 
       collect
    3. **Report:** Analyze data with VTune 

# VTune 
- In addition to the GUI, command-line tools can be used to collect the 
  statistics
    - Works with batch jobs too
- Many different profiles (actions), for example
    - *hotspots* for general overview
    - *advanced-hotspots* for more detailed view with hardware counters
    - *hpc-performance* for HPC specific analysis
    - *memory-access* for detailed memory access analysis

# VTune demo {.section}

