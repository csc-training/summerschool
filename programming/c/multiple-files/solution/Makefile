ifeq ($(COMP),)
COMP=cray
endif

ifeq ($(COMP),cray)
CC=cc
CCFLAGS=-g -I/appl/opt/libpng/include/libpng16
endif

ifeq ($(COMP),gnu)
CC=gcc
CCFLAGS=-g -Wall -I/appl/opt/libpng/include/libpng
endif


EXE=libpng
OBJS=pngwriter.o libpng.c
LIBS=-L/appl/opt/libpng/lib -lpng -lz

all: $(EXE)

pngwriter.o: pngwriter.c pngwriter.h
libpng.o: libpng.c

.PHONY: $(EXE) clean

$(EXE): $(OBJS) .ALWAYSEXEC
	$(CC) $(FCFLAGS) $(OBJS) -o $@ $(LIBS)

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o *.png

.ALWAYSEXEC:
