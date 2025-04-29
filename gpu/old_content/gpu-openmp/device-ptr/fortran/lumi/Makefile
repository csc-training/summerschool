ifeq ($(COMP),)
COMP=lumi
endif

ifeq ($(COMP),lumi)
HIPFORT_HOME = /projappl/project_465001194/apps/HIPFORT
LIB_FLAGS = -lhiprand
CXX = CC -xhip
FC = ftn -fopenmp -I$(HIPFORT_HOME)/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$(HIPFORT_HOME)/lib -lhipfort-amdgcn $(LIB_FLAGS)
endif 

OBJS=pi.o 

all: pi

pi.F90: pi.F90

pi: $(OBJS)
	$(FC) -o $@ $(OBJS) $(FCFLAGS) 

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.mod: %.F90
	$(FC) $(FCFLAGS) -c $<
clean:
	rm -f pi *.o *.mod
