CURAND_DIR=/appl/spack/v016/install-tree/gcc-4.8.5/nvhpc-21.9-qnwrcx/Linux_x86_64/21.9/math_libs/
CFLAGS=-mp=gpu -gpu=cc70 
CC=nvc

pi: pi.c
	$(CC) -o $@ pi.c $(CFLAGS) -I$(CURAND_DIR)/include -L$(CURAND_DIR)/lib64 -rpath=$(CURAND_DIR)/lib64 -lcurand 

