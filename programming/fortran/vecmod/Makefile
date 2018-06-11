ifeq ($(COMP),)
COMP=gnu
endif

ifeq ($(COMP),cray)
CC=cc
FC=ftn 
FCFLAGS=-efmw -hcpu=sandybridge
endif

ifeq ($(COMP),gnu)
FC=gfortran
FCFLAGS=-Wall
endif

ifeq ($(COMP),intel)
CC=icc
FC=ifort
endif

EXE=vec
OBJ=vecmod.o vec.o


all: $(EXE)

vecmod.o: vecmod.F90
vec.o: vec.F90 vecmod.o

$(EXE): $(OBJ)
	$(FC) $(FCFLAGS) $(OBJ) -o $@ 

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE)  *.o *.mod *~
