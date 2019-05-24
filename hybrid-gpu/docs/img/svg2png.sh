#!/bin/bash
# convert SVGs to PNGs with a correct size
inkscape data-movement.svg -e data-movement.png -w 886
inkscape execution-model.svg -e execution-model.png -w 886
inkscape memory-access.svg -e memory-access.png -w 1064
inkscape nvidia-visual-profiler.svg -e nvidia-visual-profiler.png -w 1418
inkscape vector-workers-gang.svg -e vector-workers-gang.png -w 886
