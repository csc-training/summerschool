CC=cc
HIPCC=CC -xhip

prog: prog.o util.o
	$(CC) -o $@ $^

util.o: util.cpp
	$(HIPCC) -c $<

%.o: %.cpp
	$(CC) -c $<

.PHONY: clean
clean:
	rm -f prog *.o

