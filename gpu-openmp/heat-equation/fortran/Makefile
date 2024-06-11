ifeq ($(COMP),)
COMP=lumi
endif

COMMONDIR=../common

ifeq ($(COMP),gnu)
FC=mpif90
CC=gcc
FCFLAGS=-O3 -Wall -fopenmp
CCFLAGS=-O3 -Wall -I$(COMMONDIR)
LDFLAGS=-fopenmp
LIBS=-lpng
endif

ifeq ($(COMP),nvhpc)
FC=mpif90
CC=nvc
MINFO=all
# MINFO=-Minfo=mp
# GPU_MPI=-DGPU_MPI
GPU_MPI=
FCFLAGS=-O3 -mp=gpu -gpu=cc80 $(GPU_MPI)
CCFLAGS=-O3 -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
FC=ftn
CC=cc
FCFLAGS=-O3 -fopenmp
CCFLAGS=-O3 -I$(COMMONDIR) -I$(LIBPNG_DIR)/include
LDFLAGS=-fopenmp -L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib
LIBS=-lpng
endif


EXE=heat_offload
OBJS=main.o heat_mod.o core.o setup.o utilities.o io.o pngwriter_mod.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
heat_mod.o: heat_mod.F90
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
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~  $(COMMONDIR)/*.o
