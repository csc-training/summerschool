ifneq ($(MAHTI),1)

CC=cc
CXX=CC
FLAGS=-O2 -fopenmp -I/appl/lumi/SW/LUMI-22.08/L/EB/libpng/1.6.37-cpeCray-22.08/include -L/appl/lumi/SW/LUMI-22.08/L/EB/libpng/1.6.37-cpeCray-22.08/lib

else

CC=nvc
CXX=nvc++
FLAGS=-O2 -mp=gpu -gpu=cc80 -Minfo=mp

endif

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
