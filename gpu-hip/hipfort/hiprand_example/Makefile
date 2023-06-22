ifeq ($(COMP),)
COMP=lumi
endif

LINKOPTS = -lhiprand

ifeq ($(COMP),lumi)
FC = ftn
endif 

ifeq ($(COMP),nvhpc)
# Puhti
export ROCM=${ROCM_PATH}
export HIPFORT_ARCHGPU=nvptx-sm_70
export HIPFORT_HOME=${ROCM_PATH}/hipfort/
include ${HIPFORT_HOME}/bin/Makefile.hipfort
endif


pi: hiprand_test.o
	$(FC) $(LINKOPTS)  $^ -o $@

%.o: %.F90
	$(FC) -c -o $@ $<

%.o: %.cpp
	$(CXX) -c -o $@ $<

.PHONY: clean
clean:
	rm -f *.o saxpy
