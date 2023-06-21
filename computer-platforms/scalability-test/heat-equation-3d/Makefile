ifeq ($(COMP),)
COMP=lumi
endif

COMMONDIR=.

ifeq ($(COMP),gnu)
CXX=CC
CC=gcc
HIPCC=hipcc 
HIPFLAGS=-O3 $(GPU_MEMORY)
GPU_MEMORY=-DPINNED_MEMORY 
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) -DNDEBUG -fopenmp $(GPU_MEMORY) -D__HIP_PLATFORM_AMD__
LDFLAGS=-lamdhip64
LIBS=-lpng
endif

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
CXX=CC
CC=cc
HIPCC=hipcc
GPU_MEMORY=-DPINNED_MEMORY
HIPFLAGS=$(GPU_MEMORY) -O3 -std=c++11 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -DUSE_STREAMS 
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) -I$(LIBPNG_DIR)/include -DNDEBUG $(GPU_MEMORY) -D__HIP_PLATFORM_AMD__
LDFLAGS=-L${ROCM_PATH}/lib -lamdhip64 -L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib
LIBS=-lpng
endif


EXE=heat_hip
OBJS=main.o heat.o core.o setup.o io.o utilities.o
OBJS_HIP=core_hip.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o

all: $(EXE)


utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp parallel.hpp matrix.hpp
core.o: core.cpp parallel.hpp heat.hpp
core_hip.o: core_hip.cpp heat.hpp
setup.o: setup.cpp parallel.hpp heat.hpp
io.o: io.cpp heat.hpp parallel.hpp matrix.hpp
main.o: main.cpp parallel.hpp heat.hpp

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)
$(OBJS): FLAGS := $(CCFLAGS)
$(OBJS_PNG): FLAGS := $(CCFLAGS)
$(OBJS_HIP): C_COMPILER := $(HIPCC)
$(OBJS_HIP): FLAGS := $(HIPFLAGS)


$(EXE): $(OBJS) $(OBJS_HIP) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_HIP) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(C_COMPILER) $(FLAGS) -c $< -o $@

%.o: %.c
	$(C_COMPILER) $(FLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
