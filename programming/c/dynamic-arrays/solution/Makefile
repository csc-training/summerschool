ifeq ($(COMP),)
COMP=cray
endif

COMMONDIR=../../../../heat/common

ifeq ($(COMP),cray)
CC=cc
CCFLAGS=-O3 -I/appl/opt/libpng/include
LIBS=-L/appl/opt/libpng/lib -lpng -lz
endif

ifeq ($(COMP),gnu)
CC=gcc
CCFLAGS=-O3 -Wall
LIBS=-lpng
endif

EXE=dynamic_array
OBJS=pngwriter.o dynamic_array.o

.SUFFIXES:

all: $(EXE)

pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
	$(CC) $(CCFLAGS) -I$(COMMONDIR) -c $< -o $@

dynamic_array.o: dynamic_array.c

$(EXE): $(OBJS)
	$(CC) $(CCFLAGS) $(OBJS) -o $@ $(LIBS)

%.o: %.c
	$(CC) -I$(COMMONDIR) $(CCFLAGS) -c $< -o $@

.PHONY: clean

clean:
	-/bin/rm -f $(EXE) a.out *.o *.png
