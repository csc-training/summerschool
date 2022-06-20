ifeq ($(COMP),)
COMP=gnu
endif

COMMONDIR=../../../heat/common

ifeq ($(COMP),gnu)
FC=gfortran
CC=gcc
FCFLAGS=-fopenmp -O3 -Wall
CCFLAGS=-fopenmp -O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),intel)
FC=ifort
CC=icc
FCFLAGS=-qopenmp -O3 -warn all
CCFLAGS=-qopenmp -O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

EXE=mandelbrot
OBJS=mandelbrot.o pngwriter_mod.o
OBJS_PNG=pngwriter.o
CORRECT_OBJS_PNG=$(COMMONDIR)/pngwriter.o

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
pngwriter_mod.o: pngwriter_mod.F90
mandelbrot.o: mandelbrot.F90 pngwriter_mod.o

pngwriter.o: pngwriter.c pngwriter.h
	$(CC) $(CCFLAGS) -c -o $@ $<

$(EXE): $(OBJS) $(OBJS_PNG)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~ mandelbrot.png
