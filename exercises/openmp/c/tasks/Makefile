CC=gcc
CFLAGS=-O2 -std=c99 -fopenmp
LIBS=-lpng

all: mandelbrot

pngwriter.o: pngwriter.c pngwriter.h
	gcc -O2 -c -o $@ $<

mandelbrot: mandelbrot.c pngwriter.o
	$(CC) $(CFLAGS)  $^ -o $@ $(LIBS)

clean:
	rm -f *.o mandelbrot mandelbrot.png
