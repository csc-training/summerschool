#!/bin/bash
# convert SVGs to PNGs with a correct size
inkscape affinity.svg -e affinity.png -w 709
inkscape mpi-thread-support.svg -e mpi-thread-support.png -w 1772
inkscape multiple-thread-communication.svg -e multiple-thread-communication.png -w 886
inkscape omp-parallel.svg -e omp-parallel.png -w 443
inkscape omp-summary.svg -e omp-summary.png -w 1418
inkscape processes-threads.svg -e processes-threads.png -w 1594
inkscape race-condition.svg -e race-condition.png -w 1418
inkscape reduction-operators.svg -e reduction-operators.png -w 1064
inkscape supercomputer-node-hybrid.svg -e supercomputer-node-hybrid.png -w 1772
