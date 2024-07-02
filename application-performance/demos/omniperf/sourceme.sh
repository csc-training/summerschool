#!/bin/bash

ml LUMI/23.09
ml partition/G
ml PrgEnv-cray
ml craype-accel-amd-gfx90a
ml rocm/5.4.6

export PATH=/projappl/project_465001194/apps/omniperf/bin:$PATH
