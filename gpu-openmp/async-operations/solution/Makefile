CCFLAGS=-mp=gpu -gpu=cc80
CXX=nvc++

OBJS=main.o mandelbrot.o

mandelbrot: $(OBJS)
	$(CXX) -o $@ $(CCFLAGS) $(OBJS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f mandelbrot a.out *.o *.pgm
