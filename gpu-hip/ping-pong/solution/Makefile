HIPCC = hipcc
MPICXX = mpicxx
MPICXXFLAGS = -g -O2

# Puhti
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(HIPCC) --x cu'
LDFLAGS = -L/appl/spack/install-tree/gcc-9.1.0/openmpi-4.1.1-vonyow/lib
LIBS = -lmpi

ping-pong: ping-pong.o
	$(HIPCC) $(LDFLAGS) -o $@ $< $(LIBS)

%.o: %.cpp
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f *.o ping-pong
