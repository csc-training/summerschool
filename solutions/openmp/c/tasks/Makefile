COMP=cray

ifeq ($(COMP),cray)
CC=cc
CFLAGS=-O3 -I/appl/opt/libpng/include
LDFLAGS=-L/appl/opt/libpng/lib
LIBS=-lpng -lz
endif

ifeq ($(COMP),gnu)
CC=gcc
CFLAGS=-O3 -Wall
LDFLAGS=
LIBS=-lpng
endif

all: mandelbrot

pngwriter.o: pngwriter.c pngwriter.h
	$(CC) $(CFLAGS) -c -o $@ $<

mandelbrot: mandelbrot.c pngwriter.o
	$(CC) $(CFLAGS) $(LDFLAGS)  $^ -o $@ $(LIBS)

clean:
	rm -f *.o mandelbrot mandelbrot.png
