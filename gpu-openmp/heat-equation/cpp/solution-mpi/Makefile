ifeq ($(COMP),)
COMP=nvhpc
endif


COMMONDIR=../../common

ifeq ($(COMP),gnu)
CXX=mpicxx
CC=gcc
CCFLAGS=-g -O3 -fopenmp -Wall -I$(COMMONDIR)
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

ifeq ($(COMP),intel)
CXX=icpx
CC=icx
CCFLAGS=-g -O3 -fopenmp -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

EXE=heat_hybrid
OBJS=main.o heat.o core.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


core.o: core.cpp heat.hpp
utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp 
setup.o: setup.cpp heat.hpp
io.o: io.cpp heat.hpp
main.o: main.cpp heat.hpp

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(PNG_LDFLAGS) $(LIBS) $(PNG_LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~ $(COMMDIR)/*.o
