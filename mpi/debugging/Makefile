ifeq ($(COMP),)
COMP=cray
endif


ifeq ($(COMP),gnu)
CXX=mpicxx
FC=mpif90
CCFLAGS=-g
FCFLAGS=-g
endif

ifeq ($(COMP),intel)
CXX=mpicxx
FC=mpif90
CCFLAGS=-g
FCFLAGS=-g
endif

ifeq ($(COMP),cray)
FC=ftn
CXX=CC
CCFLAGS=-g
FCFLAGS=-g
endif


all: exchange.exe collective.exe

%.exe: %.cpp
	$(CXX) $(CCFLAGS) $< -o $@

%.exe: %.F90
	$(FC) $(FCFLAGS) $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f *.exe
