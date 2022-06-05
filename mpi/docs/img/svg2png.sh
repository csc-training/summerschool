#!/bin/bash
# convert SVGs to PNGs with a correct size
for f in *svg
do
  width=`identify -format "%w\n" $f`
  outfile=`echo $f | sed -e "s/.svg/.png/"`
  inkscape $f -e $outfile -w $width
done
