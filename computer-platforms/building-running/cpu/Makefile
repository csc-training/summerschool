CC=gcc

prog: prog.o util.o
	$(CC) -o $@ $^

%.o: %.cpp
	$(CC) -c $<

.PHONY: clean
clean:
	rm -f prog *.o

