ifeq ($(COMP),)
COMP=gnu
endif

COMMONDIR=../common

ifeq ($(COMP),pgi)
CXX=mpicxx
CC=pgcc
NVCC=nvcc -ccbin pgc++
NVCCFLAGS=-g -O3 -I$(COMMONDIR)
CCFLAGS=-g -O3 -I$(COMMONDIR)
LDFLAGS=-L$(CUDA_INSTALL_ROOT)/lib64
LIBS=-lpng -lcudart
endif

ifeq ($(COMP),gnu)
CXX=mpicxx
CC=gcc
NVCC=nvcc
NVCCFLAGS=-g -O3 -I$(COMMONDIR)
CCFLAGS=-g -O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng -lcudart
endif

EXE=heat_cuda
OBJS=main.o core.o core_cuda.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


core.o: core.cpp heat.h
core_cuda.o: core_cuda.cu heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
