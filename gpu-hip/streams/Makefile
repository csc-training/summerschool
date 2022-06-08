HIPCC = hipcc
HIPFLAGS=--offload-arch=gfx908
SRC1 = async_case1.cpp
EXE1 = async_case1
SRC2 = async_case2.cpp
EXE2 = async_case2
SRC3 = async_serial.cpp
EXE3 = async_serial

.SUFFIXES:


all: $(EXE1) $(EXE2) $(EXE3)


async_case1: $(EXE1)

$(EXE1): $(SRC1)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<


async_case2: $(EXE2)

$(EXE2): $(SRC2)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<

async_case1: $(EXE3)

$(EXE3): $(SRC3)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<

.PHONY: clean

clean:
	rm -f  $(EXE1) $(EXE2) $(EXE3) 
