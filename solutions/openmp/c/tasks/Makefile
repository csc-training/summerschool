
all: mandelbrot

pngwriter.o: pngwriter.c pngwriter.h
	gcc -O2 -c -o $@ $<

mandelbrot: mandelbrot.c pngwriter.o
	gcc -O2 -std=c99 -fopenmp $^ -o $@ -lpng

clean:
	rm -f *.o mandelbrot mandelbrot.png
