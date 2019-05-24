#!/bin/bash
# convert SVGs to PNGs with a correct size
inkscape hdf5-hyperslab.svg -e hdf5-hyperslab.png -w 532
inkscape io-illustration.svg -e io-illustration.png -w 1240
inkscape io-layers.svg -e io-layers.png -w 1772
inkscape io-subarray.svg -e io-subarray.png -w 1418
inkscape lustre-architecture.svg -e lustre-architecture.png -w 1772
inkscape lustre-striping.svg -e lustre-striping.png -w 1772
inkscape nonblocking-io.svg -e nonblocking-io.png -w 1240
inkscape posix-everybody.svg -e posix-everybody.png -w 620
inkscape posix-spokesman.svg -e posix-spokesman.png -w 620
