COMP=intel

COMMONDIR=../../common
LIBPNGDIR=/appl/opt/libpng

ifeq ($(COMP),cray)
FC=ftn
CC=cc
FCFLAGS=-O3
CCFLAGS=-O3 -I$(LIBPNGDIR)/include -I$(COMMONDIR)
LDFLAGS=-L$(LIBPNGDIR)/lib
LIBS=-lpng -lz
endif

ifeq ($(COMP),gnu)
FC=mpif90
CC=gcc
FCFLAGS=-O3
CCFLAGS=-O3 -I$(LIBPNGDIR)/include -I$(COMMONDIR)
LDFLAGS=-L$(LIBPNGDIR)/lib
LIBS=-lpng -lz
endif

ifeq ($(COMP),intel)
FC=mpif90
CC=icc
FCFLAGS=-O3
CCFLAGS=-O3 -I$(LIBPNGDIR)/include -I$(COMMONDIR)
LDFLAGS=-L$(LIBPNGDIR)/lib
LIBS=-lpng -lz
endif

EXE=heat_mpi
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
