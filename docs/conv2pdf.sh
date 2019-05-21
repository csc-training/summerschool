#!/bin/bash

source_filepath=$(readlink -f $1)
print_pdf="file://$source_filepath?print-pdf#/"
filename="${1%.*}"
target_filepath="${source_filepath%.*}.pdf"

chromium-browser --headless --disable-gpu  --print-to-pdf=$target_filepath $print_pdf 

