CXX=CC
CXXFLAGS_OMP=-O2 -g -fopenmp
CXXFLAGS_HIP=-O2 -g -xhip

hip-sum.o: hip-sum.cpp
	${CXX} ${CXXFLAGS_HIP} $^ -o $@ -c

hip-interop:  hip-sum.o hip-interop.cpp
	${CXX} ${CXXFLAGS_OMP} $^ -o $@
