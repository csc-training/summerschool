#!/bin/bash

# output files
OUTPUT=ss2019-lectures.pdf
# slide of an empty page
EMPTY=empty-page.pdf
# prefix for temporary files 
#   beware: all files starting with this will be removed in the clean-up
TMP=nup-tmpfile

# filenames of lecture slides
# ??  note: 1st slide won't to used for content slides, all others will
names=(
preface.pdf
title-supercomputing.pdf
supercomputing-all.pdf
title-unix.pdf
unix-version-all.pdf
title-fortran.pdf
fortran-all.pdf
title-c.pdf
c-all.pdf
title-mpi.pdf
mpi-all.pdf
title-debugging.pdf
debugging-all.pdf
title-io.pdf
parallel-io-all.pdf
title-hybrid.pdf
hybrid-cpu-all.pdf
hybrid-gpu-all.pdf
title-performance.pdf
performance-all.pdf
title-design.pdf
design-all.pdf
epilogue.pdf
)


# prepare a A4 empty page
empty=$TMP-$EMPTY
pdfjam $EMPTY --a4paper --outfile $empty

# PRODUCE LECTURE SLIDES
queue=""
for ((i=0; i<${#names[@]}; i++))
do
	tmp=$TMP-$i.pdf
	x=$(echo ${names[$i]} | egrep '^(title-|preface.pdf|epilogue.pdf)')
	if [ "$x" != "" ]
	then
		pdfjam ${names[i]} --a4paper --outfile $tmp
	else
		# pdfjam ${names[i]} 1- --nup 2x4 --a4paper --delta '0.05cm 1.5cm' \
                pdfjam ${names[i]} --nup 2x4 --a4paper --delta '0.05cm 1.5cm' \
			--scale 0.95 --frame true --outfile $tmp

	fi
	queue="${queue} ${tmp}"
	
	# add an empty page if needed to break even
	pagecount=$(pdfinfo ${tmp} | grep Pages | cut -c8- | tr -d ' ')
	if (( (pagecount % 2) == 1 ))
	then
		queue="${queue} $empty"
	fi
done
#pdftk $queue output $OUTPUT

pdfjam --outfile $OUTPUT $queue


# remove temporary files
rm $TMP-*

