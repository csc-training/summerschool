ifeq ($(COMP),)
COMP=cray
endif

ifeq ($(COMP),cray)
CC=cc
FC=ftn 
FCFLAGS=-efmw -hcpu=sandybridge
endif

ifeq ($(COMP),gnu)
CC=gcc
FC=gfortran
FCFLAGS=-Wall
endif

ifeq ($(COMP),intel)
CC=icc
FC=ifort
endif

EXE=do-loop
OBJ=do-loop.o


all: $(EXE)

do-loop.o: do-loop.F90

$(EXE): $(OBJ)
	$(FC) $(FCFLAGS) $(OBJ) -o $@ 

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE)  a.out *.o *.mod *.png *~
