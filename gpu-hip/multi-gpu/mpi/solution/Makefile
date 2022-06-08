HIPCC = hipcc
MPICXX = mpicxx
MPICXXFLAGS = -g -O2
LD = $(HIPCC)
INCLUDES =
SRC = mpiexample.cpp
OBJS = $(SRC:.cpp=.cpp.o)
EXE = mpiexample

# Puhti
#MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(HIPCC) --x cu'
#LDFLAGS = -L/appl/spack/install-tree/gcc-9.1.0/openmpi-4.1.1-vonyow/lib
#LIBS = -lmpi

# AMD Accelerator Cloud (AAC)
MPICXXENV = MPICH_CXX='$(HIPCC)'
LDFLAGS = -L/usr/lib/x86_64-linux-gnu
LIBS = -lmpichcxx -lmpich




.SUFFIXES:


all: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBS)

mpiexample.cpp.o: mpiexample.cpp
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) $(INCLUDES) -c -o $@ $<

.PHONY: clean

clean:
	rm -f $(OBJS) *~ $(EXE) *.o
