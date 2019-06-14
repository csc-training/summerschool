#!/bin/bash

# Specify the location of slidefactory singularoty container
container=../slidefactory.simg

# Convert the .md files to html with slidefactory container and
# then to pdf with chromium
for md in title*md
do
  html="`pwd`/${md%.md}.html"
  pdf="${md%.md}.pdf"
  # echo commands
  set -x
  singularity run $container -t csc-2016-portrait --config width=1080 --config height=1527 $md
  chromium-browser --headless --print-to-pdf=$pdf file://${html}?print-pdf
  set +x
done

