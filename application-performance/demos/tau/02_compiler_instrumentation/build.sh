#!/bin/bash -l

ml LUMI/23.09
ml partition/G
ml PrgEnv-cray
ml craype-accel-amd-gfx90a
ml rocm/5.4.6

. ../sourceme.sh

export TAU_OPTIONS=-optCompInst
export TAU_MAKEFILE=$TAU_LIB/Makefile.tau-rocm-hip-rocprofiler-roctracer-cray-papi-mpi-pthread

(make -C src/hip/) || { echo "Failed to build hip code"; exit 1; }
