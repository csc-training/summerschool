CCFLAGS=-mp=gpu -gpu=cc80
CXX=nvc++
CC=nvc
# CCFLAGS=-fopenmp
# CXX=g++
# CC=gcc
CFLAGS=

LDFLAGS=-lpng

EXE=mandelbrot

all: $(EXE)

OBJS=mandelbrot.o kernel.o pngwriter.o

mandelbrot.o: mandelbrot.cpp constants.hpp

kernel.o: kernel.cpp constants.hpp

pngwrite.o: pngwriter.c pngwriter.h

$(EXE): $(OBJS)
	$(CXX) -o $@ $(CCFLAGS) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o *.png
