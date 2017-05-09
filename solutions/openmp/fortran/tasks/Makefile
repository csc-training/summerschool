FC=gfortran
FFLAGS=-O2 -fopenmp
LDFLAGS=-fopenmp
LIBS=-lpng

all: mandelbrot

pngwriter.o: pngwriter.c pngwriter.h
	gcc -O2 -c -o $@ $< 

pngwriter_mod.o: pngwriter_mod.F90
	$(FC) -c pngwriter_mod.F90

mandelbrot.o: mandelbrot.f90 pngwriter_mod.o
	$(FC) $(FFLAGS) -c mandelbrot.f90
mandelbrot: mandelbrot.o pngwriter_mod.o pngwriter.o
	$(FC) $(LDFLAGS)  -o $@ mandelbrot.o pngwriter_mod.o pngwriter.o $(LIBS)

clean:
	rm -f *.o *.mod mandelbrot mandelbrot.png
