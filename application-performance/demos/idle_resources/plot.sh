#!/bin/bash

ml LUMI/24.03
ml partition/C
ml gnuplot/5.4.10-cpeGNU-24.03

gnuplot plot_runtimes.gnuplot
