ifeq ($(COMP),)
COMP=gnu
endif

COMMONDIR=../common

ifeq ($(COMP),gnu)
CXX=mpicxx
CC=gcc
HIPCC=hipcc --x cu
HIPFLAGS=-g -O3 -I$(COMMONDIR)
CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) 
LDFLAGS=
LIBS=-lpng -lamdhip64 -lcudart
endif

ifeq ($(COMP),lumi)
CXX=mpicxx
CC=gcc
HIPCC=hipcc --x cu
HIPFLAGS=-g -O3 -I$(COMMONDIR)
CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) 
LDFLAGS=
LIBS=-lpng -lamdhip64 -lcudart
endif

EXE=heat_hip
OBJS=main.o core.o setup.o utilities.o io.o 
OBJS_PNG=$(COMMONDIR)/pngwriter.o
OBJS_HIP=core_hip.o


all: $(EXE)


core.o: core.cpp heat.h
core_hip.o: core_hip.cpp heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)
$(OBJS): FLAGS := $(CCFLAGS)
$(OBJS_HIP): C_COMPILER := $(HIPCC)
$(OBJS_HIP): FLAGS := $(HIPFLAGS)

$(EXE): $(OBJS) $(OBJS_HIP) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_HIP) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(C_COMPILER) $(FLAGS) -c $< -o $@

%.o: %.cu
	$(C_COMPILER) $(FLAGS) -c $< -o $@


%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
