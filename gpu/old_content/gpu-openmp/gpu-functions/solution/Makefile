ifeq ($(COMP),)
COMP=lumi
endif

ifeq ($(COMP),gnu)
FC=mpif90
CC=gcc
FCFLAGS=-O3 -Wall -fopenmp
CCFLAGS=-O3 -Wall -fopenmp
endif

ifeq ($(COMP),nvhpc)
FC=nvfortran
CC=nvc
FCFLAGS=-O3 -mp=gpu -gpu=cc80 -Minfo=mp
CCFLAGS=-O3 -mp=gpu -gpu=cc80 -Minfo=mp
endif

ifeq ($(COMP),lumi)
FC=ftn
CC=cc
FCFLAGS=-O3 -fopenmp
CCFLAGS=-O3 -fopenmp
endif

all: c fortran

c: sum_c

fortran: sum_ftn

sum_c: my_sum.o sum_c.o
	$(CC) $(CCFLAGS) sum_c.o my_sum.o -o $@

sum_ftn: my_mod.o sum_ftn.o
	$(FC) $(FCFLAGS) sum_ftn.o my_mod.o -o $@

sum_c.o: sum.c
	$(CC) $(CCFLAGS) -c $< -o $@

sum_ftn.o: sum.F90
	$(FC) $(FCFLAGS) -c $< -o $@

my_mod.o: my_mod.F90
	$(FC) $(FCFLAGS) -c $< -o $@

my_sum.o: my_sum.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f sum_c sum_ftn a.out *.o *.mod *.s *~
