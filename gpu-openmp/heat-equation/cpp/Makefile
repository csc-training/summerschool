ifeq ($(COMP),)
COMP=lumi
endif

COMMONDIR=../common

ifeq ($(COMP),gnu)
CXX=mpicxx
CC=gcc
CCFLAGS=-g -O3 -fopenmp -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),intel)
CXX=icpx
CC=icx
CCFLAGS=-g -O3 -fopenmp -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),nvhpc)
CXX=mpicxx
CC=nvc
MINFO=
# MINFO=-Minfo=mp
#GPU_MPI=-DGPU_MPI
GPU_MPI=
CCFLAGS=-O3 -mp=gpu -gpu=cc70 -I$(COMMONDIR) $(MINFO) $(GPU_MPI)
# CCFLAGS=-O3 -I$(COMMONDIR) $(MINFO)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
CXX=CC
CC=cc
CCFLAGS=-O3 -fopenmp -Wall -I$(COMMONDIR) -I$(LIBPNG_DIR)/include
LDFLAGS=-L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib
LIBS=-lpng
endif


EXE=heat_offload
OBJS=main.o heat.o core.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)

utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp 
core.o: core.cpp heat.hpp
setup.o: setup.cpp heat.hpp
io.o: io.cpp heat.hpp
main.o: main.cpp heat.hpp

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~ $(COMMDIR)/*.o
