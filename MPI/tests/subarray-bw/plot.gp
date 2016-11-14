set terminal png size 1024,768 fontscale 2
set output 'out.png'

set key left top
set logscale x
#set logscale y
set xrange [1:2100000]
set xlabel 'Message size (Bytes)'
set ylabel 'GB/s'
plot 'man.dat' u 4:6 w lp lw 3 t 'Manual packing', \
     'trans.dat' u 4:6 w lp lw 3 t 'Subarray',\
     'all.dat' u 4:6 w lp lw 3 t 'Subarray, including commit'
     
     
