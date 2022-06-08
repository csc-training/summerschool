SOURCES=$(wildcard *.md)
TARGETS=$(patsubst %.md,%.html,$(SOURCES))
PDFS=$(patsubst %.md,%.pdf,$(SOURCES))

SINGULARITY=singularity
SARGS=run
SIMAGE=../../slide-template/slidefactory.simg
OPTIONS=--theme=csc-2016-prace

all: $(TARGETS)

pdf: $(PDFS)

%.html: %.md
	$(SINGULARITY) $(SARGS) $(SIMAGE) $(OPTIONS) $<

%.pdf: %.html
	chromium-browser --headless --print-to-pdf=$@ file://`pwd`/$(<)?print-pdf

.PHONY: clean
clean:
	-/bin/rm -f *.html *.pdf *~
