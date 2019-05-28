SOURCES=$(wildcard *.md)
TARGETS=$(patsubst %.md,%.html,$(SOURCES))

SINGULARITY=singularity
SARGS=run
SIMAGE=../../docs/slide-template/slidefactory.simg

all: $(TARGETS)

%.html: %.md
	$(SINGULARITY) $(SARGS) $(SIMAGE) $<

.PHONY: clean
clean:
	-/bin/rm -f *.html *~
