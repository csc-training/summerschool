ifeq ($(COMP),)
COMP=lumi
endif

ifeq ($(COMP),lumi)
HIPRAND_DIR=$(ROCM_PATH)/hiprand
CFLAGS=-D__HIP_PLATFORM_AMD__ -O2 -fopenmp -std=c++14
LIBS=-I$(HIPRAND_DIR)/include -L$(HIPRAND_DIR)/lib  -lhiprand 
CC=CC
endif


ifeq ($(COMP),nvhpc)
CURAND_DIR=/appl/spack/v016/install-tree/gcc-4.8.5/nvhpc-21.9-qnwrcx/Linux_x86_64/21.9/math_libs/
CFLAGS=-mp=gpu -gpu=cc70
LIBS=-I$(CURAND_DIR)/include -L$(CURAND_DIR)/lib64 -rpath=$(CURAND_DIR)/lib64 -lcurand  
CC=nvc
endif

pi: pi.c
	$(CC) -o $@ pi.c $(CFLAGS)  $(LIBS) 


clean:
	-rm -f pi
