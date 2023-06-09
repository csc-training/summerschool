ifeq ($(COMP),)
COMP=intel
endif

COMMONDIR=../../common

ifeq ($(COMP),intel)
FC=ifort
CC=icc
FCFLAGS=-O2 -xHost
CCFLAGS=-O2 -xHost -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),gnu)
FC=gfortran
CC=gcc
FCFLAGS=-O3 -Wall -march=native
CCFLAGS=-O3 -Wall -march=native -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

EXE=heat
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
