ifneq ($(PUHTI),1)

CC=cc
FC=ftn
FLAGS=-O2 -fopenmp

else

CC=nvc
FC=nvfortran
FLAGS=-O2 -mp=gpu -gpu=cc70

endif

LDFLAGS=-lpng

EXE=mandelbrot

OBJS=mandelbrot.o constants.o kernel.o pngwriter_mod.o pngwriter.o

all: $(EXE)

mandelbrot.o: mandelbrot.F90 kernel.o pngwriter_mod.o

kernel.o: kernel.F90 constants.o

mandelbrot: $(OBJS)
	$(CC) -o $@ $(FLAGS) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *.s
