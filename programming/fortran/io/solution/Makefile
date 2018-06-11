ifeq ($(COMP),)
COMP=gnu
endif

COMMONDIR=../../../../heat/common
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
FC=gfortran
CC=gcc
FCFLAGS=-O3 -Wall 
CCFLAGS=-O3 -Wall -I$(LIBPNGDIR)/include -I$(COMMONDIR)
LDFLAGS=-L$(LIBPNGDIR)/lib
LIBS=-lpng -lz
endif

EXE=io
OBJS=main.o io.o pngwriter_mod.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
io.o: io.F90 pngwriter_mod.o 
pngwriter_mod.o: pngwriter_mod.F90
main.o: main.F90 io.o

$(EXE): $(OBJS) $(OBJS_PNG)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
