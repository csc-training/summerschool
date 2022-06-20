FCFLAGS=-mp=gpu -gpu=cc80
FC=nvfortran
CC=nvc
#FCFLAGS=-g -fopenmp -O0
#FC=gfortran
#CC=gcc

LDFLAGS=-lpng

EXE=mandelbrot

OBJS=mandelbrot.o constants.o kernel.o pngwriter_mod.o pngwriter.o

all: $(EXE)

mandelbrot.o: mandelbrot.F90 kernel.o pngwriter_mod.o

kernel.o: kernel.F90 constants.o

mandelbrot: $(OBJS)
	$(FC) -o $@ $(FCFLAGS) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png
