#!/bin/bash

# Specify the location of slidefactory singularoty container
container=../slidefactory.simg

md_files=(
../intro-to-hpc/docs/supercomputing-all.md
../unix-version-control/docs/unix-version-all.md
../programming/fortran/docs/fortran-all.md
../programming/c/docs/c-all.md
../mpi/docs/mpi-all.md
../debugging/docs/debugging-all.md
../parallel-io/docs/parallel-io-all.md
../hybrid/docs/hybrid-cpu-all.md
../hybrid-gpu/docs/hybrid-gpu-all.md
../application-performance/docs/performance-all.md
../application-design/docs/design-all.md
)

# Convert the .md files to html with slidefactory container and
# then to pdf with chromium
for md in ${md_files[*]}
do
  html="`pwd`/${md%.md}.html"
  pdf="$(basename ${md%.md}.pdf)"
  # echo commands
  set -x
  singularity run $container $md
  chromium-browser --headless --print-to-pdf=$pdf file://${html}?print-pdf
  set +x
done

