ifeq ($(COMP),)
COMP=lumi
endif

COMMONDIR=../common

ifeq ($(COMP),intel)
CC=mpicc
CCFLAGS=-O3 -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),gnu)
CC=mpicc
CCFLAGS=-O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
CC=cc
CCFLAGS=-O3 -Wall -I$(COMMONDIR) -I$(LIBPNG_DIR)/include
LDFLAGS=-L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib
LIBS=-lpng
endif


EXE=heat_mpi
OBJS=core.o setup.o utilities.o io.o main.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
core.o: core.c heat.h
utilities.o: utilities.c heat.h
setup.o: setup.c heat.h
io.o: io.c heat.h
main.o: main.c heat.h

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CC)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CC) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.c
	$(C_COMPILER) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
