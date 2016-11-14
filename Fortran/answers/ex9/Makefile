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

EXE=heat
OBJS=heat.o main.o
OBJS_PNG=pngwriter.o
OBJS_PNGF=pngwriter_mod.o 


all: $(EXE) 
pngwriter.o: pngwriter.c
pngwriter_mod.o: pngwriter_mod.F90
heat.o: heat.F90 pngwriter_mod.o pngwriter.o
main.o: main.F90 heat.F90

$(OBJS): F_COMPILER := $(FC)
$(OBJS_PNGF): F_COMPILER := $(FC)


$(EXE): $(OBJS) $(OBJS_PNG) $(OBJS_PNGF)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) $(OBJS_PNGF) -o $@ $(LIBS)


%.o: %.F90
	$(F_COMPILER) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
