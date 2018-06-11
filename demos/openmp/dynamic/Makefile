ifeq ($(COMP),)
COMP=cray
endif

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
FCFLAGS=-fopenmp -O3 -Wall
CCFLAGS=-fopenmp -O3 -Wall
LDFLAGS=
LIBS=-lpng
endif

EXE=mandelbrot
OBJS=mandelbrot.o
OBJS_PNG=pngwriter.o

all: $(EXE)

pngwriter.o: pngwriter.c pngwriter.h
mandelbrot.o: mandelbrot.c

pngwriter.o: pngwriter.c pngwriter.h
	$(CC) $(CCFLAGS) -c -o $@ $<

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CC) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~ mandelbrot.png
