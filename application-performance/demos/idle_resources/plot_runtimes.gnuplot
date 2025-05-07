# Common settings

set terminal png size 2000,2000 font "default,50";
set style data linespoints;
set key left top;
set grid;

set logscale x;
set logscale y;
set xlabel "problem size";
set ylabel "time [ns]";
set xrange [10:10000000000];
set yrange [10:10000000000];
set xtics rotate by 45 right;

# Plot all
set output "runtimes_all.png";
set title "Runtime of Taylor expansion of e^x for different N";

plot "serial_0.dat"   title "serial N = 0"  lw 3.0 pt 5 ps 4,\
     "serial_4.dat"   title "serial N = 4"  lw 3.0 pt 5 ps 4,\
     "serial_8.dat"   title "serial N = 8"  lw 3.0 pt 5 ps 4,\
     "serial_12.dat"  title "serial N = 12" lw 3.0 pt 5 ps 4,\
     "serial_16.dat"  title "serial N = 16" lw 3.0 pt 5 ps 4,\
     "omp_0.dat"      title "OMP64 N = 0"   lw 3.0 pt 7 ps 4,\
     "omp_4.dat"      title "OMP64 N = 4"   lw 3.0 pt 7 ps 4,\
     "omp_8.dat"      title "OMP64 N = 8"   lw 3.0 pt 7 ps 4,\
     "omp_12.dat"     title "OMP64 N = 12"  lw 3.0 pt 7 ps 4,\
     "omp_16.dat"     title "OMP64 N = 16"  lw 3.0 pt 7 ps 4,\
     "hip_0.dat"      title "gpu N = 0"     lw 3.0 pt 9 ps 4,\
     "hip_4.dat"      title "gpu N = 4"     lw 3.0 pt 9 ps 4,\
     "hip_8.dat"      title "gpu N = 8"     lw 3.0 pt 9 ps 4,\
     "hip_12.dat"     title "gpu N = 12"    lw 3.0 pt 9 ps 4,\
     "hip_16.dat"     title "gpu N = 16"    lw 3.0 pt 9 ps 4;

# N = 0
set output "runtimes_0.png"; 
set title "Runtime of Taylor expansion of e^x for N = 0";

plot "serial_0.dat"   title "serial N = 0"  lw 3.0 pt 5 ps 4,\
     "omp_0.dat"      title "OMP64 N = 0"   lw 3.0 pt 7 ps 4,\
     "hip_0.dat"      title "gpu N = 0"     lw 3.0 pt 9 ps 4;

# N = 16
set output "runtimes_16.png"; 
set title "Runtime of Taylor expansion of e^x for N = 16";

plot "serial_16.dat"   title "serial N = 16"  lw 3.0 pt 5 ps 4,\
     "omp_16.dat"      title "OMP64 N = 16"   lw 3.0 pt 7 ps 4,\
     "hip_16.dat"      title "gpu N = 16"     lw 3.0 pt 9 ps 4;
