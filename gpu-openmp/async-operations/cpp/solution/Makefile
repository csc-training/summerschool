CC=cc
CXX=CC
FLAGS=-O2 -fopenmp

LDFLAGS=-lpng

EXE=mandelbrot

all: $(EXE)

OBJS=mandelbrot.o kernel.o pngwriter.o

mandelbrot.o: mandelbrot.cpp constants.hpp

kernel.o: kernel.cpp constants.hpp

pngwrite.o: pngwriter.c pngwriter.h

$(EXE): $(OBJS)
	$(CXX) -o $@ $(FLAGS) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o *.png
