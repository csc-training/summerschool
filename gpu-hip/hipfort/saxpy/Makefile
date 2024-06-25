ifeq ($(COMP),)
COMP=lumi
endif

ifeq ($(COMP),lumi)
HIPFORT_HOME = /projappl/project_465001194/apps/HIPFORT
CXX = CC -xhip
FC = ftn -I$(HIPFORT_HOME)/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$(HIPFORT_HOME)/lib -lhipfort-amdgcn $(LIB_FLAGS)
endif 

ifeq ($(COMP),nvhpc)
# Puhti
export ROCM=${ROCM_PATH}
export HIPFORT_ARCHGPU=nvptx-sm_70
export HIPFORT_HOME=${ROCM_PATH}/hipfort/
include ${HIPFORT_HOME}/bin/Makefile.hipfort
CXX = $(HIPCC_ENV) $(HIP_PATH)/bin/hipcc -x cu $(HIPCC_OPTS)
endif

saxpy: main.o hipsaxpy.o
	$(FC) $^ $(LINKOPTS) -o $@

%.o: %.f03
	$(FC) -c -o $@ $<

%.o: %.cpp
	$(CXX) -c -o $@ $<

.PHONY: clean
clean:
	rm -f *.o saxpy
