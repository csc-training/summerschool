COMP=gnu

COMMONDIR=./

ifeq ($(COMP),intel)
FC=gfortran
CC=mpiicc
FCFLAGS= -O3 -Wall 
CCFLAGS= -O3 -Wall -std=c99 -I$(COMMONDIR) -tcollect -trace
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),gnu)
FC=gfortran
CC=mpicc
FCFLAGS= -O3 -Wall 
CCFLAGS= -O3 -Wall -std=c99 -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

EXE=mandelbrot
OBJS=mandelbrot.o
OBJS_PNG=pngwriter.o
CORRECT_OBJS_PNG=$(COMMONDIR)/pngwriter.o

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
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
