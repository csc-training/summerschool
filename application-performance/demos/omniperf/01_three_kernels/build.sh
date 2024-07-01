#!/bin/bash

ml LUMI/23.09
ml partition/G
ml rocm/5.4.6
ml PrgEnv-cray/8.4.0

CC -xhip -pg -O2 main.cpp
