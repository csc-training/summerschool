#!/bin/bash
# file containing a list of input files (@ marks A4 PDFs == include as-is)
index=index.jam
# output file name
output=lecture-slides.pdf

# default page selections
pages_first='2-'   # first file in a section
pages_normal='2-'  # all other files

# prefix for temp files (be careful!)
tmp=.tmp

# joins all PDF pages (files and page selections given as arguments)
# and then jams multiple pages into a single page
jam() {
    local output=$1
    shift
    local args="$@"
    joined=$tmp-joined.pdf

    pdfjam --fitpaper true --rotateoversize true --outfile $joined -- $args
    pdfjam --nup 2x4 --a4paper --delta '0.05cm 1.5cm' --scale 0.95 \
        --frame true --outfile $output -- $joined 1-
}
# queues a PDF to the final output (and adds an empty page if needed)
add_to_manifest() {
    local file=$1

    local pagecount=$(pdfinfo $file | grep Pages | cut -c8- | tr -d ' ')
    local spec="$out 1-"
    if (( (pagecount % 2) == 1 ))
    then
        spec="${spec},{}"
    fi
    manifest="$manifest $spec"
}

manifest=""
todo=""
o=0
for name in $(grep -v "^#" $index)
do
    echo "> $name"
    # check for title slides
    title=0
    if [ -z "${name##@*}" ]
    then
        title=1
        name=${name:1}
    fi

    if (( $title ))
    then
        # check if previous section needs to be processed first
        if [[ $todo != "" ]]
        then
            out=$tmp-${o}.pdf
            let "o++"
            jam $out $todo
            add_to_manifest $out
            todo=""
        fi
        # include the entire A4 PDF as it is
        out=$tmp-$(basename $name)
        pdfjam --fitpaper true --rotateoversize true $name --outfile $out
        add_to_manifest $out
    else
        # select correct page range and add the file to the TODO list
        if [[ $todo != "" ]]
        then
            pages=$pages_normal
        else
            pages=$pages_first
        fi
        todo="$todo $name $pages"
    fi
done
# check if previous section needs to be processed still
if [[ "$todo" != "" ]]
then
    out=$tmp-${o}.pdf
    let "o++"
    jam $out $todo
    add_to_manifest $out
fi

# jam it all together
echo ">> $manifest"
pdfjam $manifest --outfile $output

# remove temp files
rm ${tmp}*
