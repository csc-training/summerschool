ifeq ($(COMP),)
COMP=lumi
endif

COMMONDIR=../common

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
CXX=CC
CC=cc
CXXFLAGS=-xhip -I$(COMMONDIR)
CCFLAGS=-I$(LIBPNG_DIR)/include
LDFLAGS=-L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib
LIBS=-lpng
endif

EXE=heat_hip
OBJS=main.o core.o core_hip.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


core.o: core.cpp heat.h
core_hip.o: core_hip.cpp heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
