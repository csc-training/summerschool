COMP=cray

ifeq ($(COMP),cray)
CC=cc
CCFLAGS=-O3 -I/appl/opt/libpng/include/libpng16
LIBS=-L/appl/opt/libpng/lib -lpng -lz
endif

ifeq ($(COMP),gnu)
CC=mpicc
CCFLAGS=-O3 -Wall
LIBS=-lpng
endif

EXE=heat
OBJS=pngwriter.o heat.o main.o

all: $(EXE)

pngwriter.o: pngwriter.c pngwriter.h
heat.o: heat.c heat.h
main.o: main.c heat.h

$(EXE): $(OBJS)
	$(CC) $(CCFLAGS) $(OBJS) -o $@ $(LIBS)

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o *.png
