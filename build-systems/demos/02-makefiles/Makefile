
.PHONY: clean all install

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

hello: hello.o module.o

clean:
	rm *.o hello

install: hello
	mkdir -p ./bin/
	cp hello ./bin/

all: hello
