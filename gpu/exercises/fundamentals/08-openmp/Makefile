CXX=CC
CXXFLAGS_OMP=-O2 -g -fopenmp
CXXFLAGS_HIP=-O2 -g -xhip

hip-sum.o: hip-sum.cpp
	${CXX} ${CXXFLAGS_HIP} $^ -o $@ -c

01-vec-add:  hip-sum.o 01-vec-add.cpp
	${CXX} ${CXXFLAGS_OMP} $^ -o $@
