HIPCC = hipcc
HIPFLAGS=--offload-arch=gfx908
SRC1 = copy.cpp
EXE1 = copy
SRC2 = matrix_transpose_naive.cpp
EXE2 = matrix_transpose_naive
SRC3 = matrix_transpose_lds.cpp
EXE3 = matrix_transpose_lds

.SUFFIXES:


all: $(EXE1) $(EXE2)


copy: $(EXE1)

$(EXE1): $(SRC1)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<


matrix_transpose_naive: $(EXE2)

$(EXE2): $(SRC2)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<


matrix_transpose_lds: $(EXE3)

$(EXE3): $(SRC3)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<

.PHONY: clean

clean:
	rm -f  $(EXE1) $(EXE2) 
