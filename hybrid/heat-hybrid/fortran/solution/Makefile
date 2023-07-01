ifeq ($(COMP),)
COMP=lumi
endif

COMMONDIR=../../common

ifeq ($(COMP),intel)
FC=mpif90
CC=icc
FCFLAGS=-O2 -xHost -fopenmp
CCFLAGS=-O2 -xHost -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),gnu)
FC=mpif90
CC=gcc
FCFLAGS=-O3 -Wall -march=native -fopenmp
CCFLAGS=-O3 -Wall -march=native -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
FC=ftn
CC=cc
FCFLAGS=-O3 -fopenmp
CCFLAGS=-O3 -I$(COMMONDIR) -I$(LIBPNG_DIR)/include -fopenmp
LDFLAGS=-L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib -fopenmp
LIBS=-lpng
endif

EXE=heat_hybrid
OBJS=main.o heat_mod.o core.o setup.o utilities.o io.o pngwriter_mod.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
core.o: core.F90 heat_mod.o
utilities.o: utilities.F90 heat_mod.o
io.o: io.F90 heat_mod.o pngwriter_mod.o 
setup.o: setup.F90 heat_mod.o utilities.o io.o
pngwriter_mod.o: pngwriter_mod.F90 heat_mod.o
main.o: main.F90 heat_mod.o core.o io.o setup.o utilities.o

$(EXE): $(OBJS) $(OBJS_PNG)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
