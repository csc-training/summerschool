COMP=gnu

ifeq ($(COMP),cray)
CXX=c
CXXFLAGS=-O3 -I/appl/opt/libpng/include
LIBS=-L/appl/opt/libpng/lib -lpng -lz
endif

ifeq ($(COMP),gnu)
CXX=c
CXX=g++
CCFLAGS=-O2 -Wall
CXXFLAGS=-O2 -Wall 
LIBS=-lpng 
endif

EXE=array2d
OBJS=array2d.o pngwriter.o

all: $(EXE)


$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LIBS)

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -std=c++14 -c $< -o $@

clean:
	-/bin/rm -f $(EXE) a.out *.o
