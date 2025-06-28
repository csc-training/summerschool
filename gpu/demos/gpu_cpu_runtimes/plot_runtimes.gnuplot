# Common settings

set terminal png size 3840,2160 font "default,90";
set style data linespoints;

set lmargin at screen 0.085;
set rmargin at screen 0.997;
set tmargin at screen 0.997;
set bmargin at screen 0.14;
#set rmargin 14;

#set key rmargin;
set key top left;
set border lw 8.0;
set grid;

set logscale x;
set logscale y;
set xlabel "vector size [10^x]" offset screen 0.0, 0.03;
set ylabel "time [10^y ns]" offset screen 0.03, 0.0;
set xrange [2e1:3e9];
set yrange [1e2:1e11];
#set xtics rotate by 45 right;
set xtics ("2" 1e2, "3" 1e3, "4" 1e4, "5" 1e5, "6" 1e6, "7" 1e7, "8" 1e8, "9" 1e9);
set ytics ("2" 1e2, "3" 1e3, "4" 1e4, "5" 1e5, "6" 1e6, "7" 1e7, "8" 1e8, "9" 1e9, "10" 1e10);

# N = 0
set output "runtimes_0.png"; 
#set title "Runtime of Taylor expansion of e^x for N = 0";

plot "serial_0.dat"   title "ser 0"   lw 10.0 pt 5 ps 4,\
     "omp_0.dat"      title "omp 0"   lw 10.0 pt 7 ps 4,\
     "hip_0.dat"      title "gpu 0"   lw 10.0 pt 9 ps 4;

# N = 16
set output "runtimes_16.png"; 
#set title "Runtime of Taylor expansion of e^x for N = 16";

plot "serial_16.dat"   title "ser 16"   lw 10.0 pt 5 ps 4,\
     "omp_16.dat"      title "omp 16"   lw 10.0 pt 7 ps 4,\
     "hip_16.dat"      title "gpu 16"   lw 10.0 pt 9 ps 4;

# Plot all
set output "runtimes_all.png";
#set title "Runtime of Taylor expansion of e^x for different N";

plot "serial_0.dat"   title "ser 0"     lw 10.0 pt 5 ps 4,\
     "serial_8.dat"   title "ser 8"     lw 10.0 pt 5 ps 4,\
     "serial_16.dat"  title "ser 16"    lw 10.0 pt 5 ps 4,\
     "omp_0.dat"      title "omp 0"     lw 10.0 pt 7 ps 4,\
     "omp_8.dat"      title "omp 8"     lw 10.0 pt 7 ps 4,\
     "omp_16.dat"     title "omp 16"    lw 10.0 pt 7 ps 4,\
     "hip_0.dat"      title "gpu 0"     lw 10.0 pt 9 ps 4,\
     "hip_8.dat"      title "gpu 8"     lw 10.0 pt 9 ps 4,\
     "hip_16.dat"     title "gpu 16"    lw 10.0 pt 9 ps 4;

