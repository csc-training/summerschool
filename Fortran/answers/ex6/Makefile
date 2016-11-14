COMP=cray

ifeq ($(COMP),cray)
CC=cc
FC=ftn 
CCFLAGS=-hcpu=sandybridge -I/appl/opt/libpng/include/libpng16
FCFLAGS=-efmw -hcpu=sandybridge
LIBS=-L/appl/opt/libpng/lib -lpng16 -lz 
endif

ifeq ($(COMP),gnu)
CC=gcc
FC=gfortran
CCFLAGS=-Wall -I/appl/opt/libpng/include/libpng16
FCFLAGS=-Wall
MPIF90FLAGS=-Wall
LIBS=-L/appl/opt/libpng/lib -lpng16 -lz
endif

ifeq ($(COMP),intel)
CC=icc
FC=ifort
CCFLAGS=-I/appl/opt/libpng/include/libpng16
FCFLAGS=
LIBS=-L/appl/opt/libpng/lib -lpng16 -lz
endif

EXE=io
OBJS=pngwriter.o io.o

all:    $(EXE)

io.o: pngwriter.o io.F90


$(OBJS): F_COMPILER := $(FC)


$(EXE): $(OBJS) 
	$(FC) $(FCFLAGS) $(OBJS) -o $@ $(LIBS)


%.o: %.F90
	$(F_COMPILER) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
