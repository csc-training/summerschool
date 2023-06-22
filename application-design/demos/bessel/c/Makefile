default: build
	echo "Start Build"

# Accelerator architecture
ifeq ($(HOST),1)

CXX = gcc
CXXFLAGS = -g -O3
FILETYPE = .c
EXE = bessel

else ifeq ($(HIP),CUDA)

CXX = hipcc
CXXDEFS = -DHAVE_HIP -I/appl/opt/rocm/rocm-4.0.0/hiprand/include/
CXXFLAGS = -g -O3 --x=cu --extended-lambda -gencode=arch=compute_70,code=sm_70
# CXXFLAGS = -g -O3 -Xcompiler -fno-tree-vectorize -Xcompiler -fopt-info-loop --x=cu --extended-lambda
FILETYPE = .cpp
EXE = bessel

else ifeq ($(HIP),ROCM)

CXX = hipcc
CXXDEFS = -DHAVE_HIP
CXXFLAGS = -g -O3
FILETYPE = .cpp
EXE = bessel

else

CXX = nvcc
CXXDEFS = -DHAVE_CUDA
CXXFLAGS = -g -O3 --x=cu --extended-lambda -gencode=arch=compute_70,code=sm_70
FILETYPE = .c
EXE = bessel

endif

# Message passing protocol
ifeq ($(MPI),1)

MPICXX = mpicxx
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(CXX) -DHAVE_MPI $(CXXDEFS) $(CXXFLAGS)'
LDFLAGS = -L/appl/spack/install-tree/gcc-9.1.0/openmpi-4.1.1-vonyow/lib
LIBS = -lmpi -lm

else

MPICXX = $(CXX)
MPICXXFLAGS = $(CXXDEFS) $(CXXFLAGS)
LIBS = -lm

endif

# Create temporary .cpp files if needed (for HIP only)
ifeq ($(FILETYPE),.cpp)
$(shell for file in `ls src/*.c`;\
		do cp -- "$$file" "$${file%.c}.cpp";\
		done)
endif

# Identify sources and objects
SRC_PATH = src/
SOURCES = $(shell ls src/*$(FILETYPE))

OBJ_PATH = src/
OBJECTS = $(shell for file in $(SOURCES);\
		do echo -n $$file | sed -e "s/\(.*\)\$(FILETYPE)/\1\.o/";echo -n " ";\
		done)

build: $(EXE)

depend:
	makedepend $(CXXDEFS) -m $(SOURCES)

test: $(EXE)
	./$(EXE)

# KOKKOS_DEFINITIONS are outputs from Makefile.kokkos 
# see https://github.com/kokkos/kokkos/wiki/Compiling
$(EXE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $(EXE)

clean: $(CLEAN)
	rm -f $(OBJECTS) $(EXE) src/*.cpp

# Compilation rules
$(OBJ_PATH)%.o: $(SRC_PATH)%$(FILETYPE)
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c $< -o $(SRC_PATH)$(notdir $@)
