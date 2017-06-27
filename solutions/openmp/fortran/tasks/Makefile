COMP=cray

ifeq ($(COMP),cray)
FC=ftn
CC=cc
FCFLAGS=-O3
CCFLAGS=-O3 -I/appl/opt/libpng/include 
LDFLAGS=-L/appl/opt/libpng/lib
LIBS=-lpng -lz
endif

ifeq ($(COMP),gnu)
FC=gfortran
CC=gcc
FCFLAGS=-O3 -Wall
CCFLAGS=-O3 -Wall 
LDFLAGS=
LIBS=-lpng
endif

all: mandelbrot

pngwriter.o: pngwriter.c pngwriter.h
	$(CC) $(CCFLAGS) -c -o $@ $< 

pngwriter_mod.o: pngwriter_mod.F90
	$(FC) -c pngwriter_mod.F90

mandelbrot.o: mandelbrot.f90 pngwriter_mod.o
	$(FC) $(FFLAGS) -c mandelbrot.f90
mandelbrot: mandelbrot.o pngwriter_mod.o pngwriter.o
	$(FC) $(LDFLAGS)  -o $@ mandelbrot.o pngwriter_mod.o pngwriter.o $(LIBS)

clean:
	rm -f *.o *.mod mandelbrot mandelbrot.png
