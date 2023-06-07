#!/usr/local/bin/gnuplot -persist
set terminal pngcairo  transparent enhanced font "arial,30" fontscale 1.0 size 1920, 1080 
set output 'gpu_approaches.png'
unset key
unset parametric
set view map scale 1
set style data lines
set xtics border in scale 0,0 mirror norotate  autojustify
set xtics  norangelimit 
set xtics   ()
set ytics border in scale 0,0 mirror norotate  autojustify
set ytics  norangelimit 
set ytics   ()
set ztics border in scale 0,0 nomirror norotate  autojustify
unset cbtics
set rtics axis in scale 0,0 nomirror norotate  autojustify
#set title "Programming GPUs: different approaches" 
set xrange [ -0.500000 : 3.50000 ] noreverse nowriteback
set x2range [ * : * ] noreverse writeback
set yrange [ -0.500000 : 3.50000 ] noreverse nowriteback
set y2range [ * : * ] noreverse writeback
set zrange [ * : * ] noreverse writeback
set cbrange [ 0.00000 : 5.00000 ] noreverse nowriteback
set rrange [ * : * ] noreverse writeback
#set palette model RGB defined ( 0 0 1 0, 0.3333 0 0.3922 0, 0.3333 1 1 0, 0.6667 0.7843 0.7843 0, 0.6667 1 0 0, 1 0.5451 0 0 )
#set palette defined ( 1 '#FFB000',2 '#FE6100' , 3 '#DC267F',4 '#785EF0' ,5 '#648FFF')
set palette defined ( 1 '#fc7f00',2 '#ffa000' , 3 '#ffbd00',4 '#ffe81a' ,5 '#e4ff7a')
set colorbox vertical origin screen 0.9, 0.2 size screen 0.05, 0.6 front  noinvert bdefault
NO_ANIMATION = 1
## Last datafile plotted: "$map3"
#plot '$map3' matrix rowheaders columnheaders using 1:2:3 with image

$map3 << EOD
,Ease of Use,Portability,Performances,Availability
   Native Languages, 1, 1, 5, 5
Directive Languages, 2, 3, 3, 4
      GPU Libraries, 4, 3, 5, 3
    GPU Application, 5, 2, 4, 1
EOD

set datafile separator comma
set cbtics ("bad" 0, "good" 5)
plot '$map3' matrix rowheaders columnheaders using 1:2:3 with image
set datafile separator



