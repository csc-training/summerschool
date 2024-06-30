FACTORY=slidefactory.sif
OPTIONS=

SRC=$(wildcard *.md)
HTML=$(patsubst %.md,%.html,$(SRC))
PDF=$(patsubst %.md,%.pdf,$(SRC))

.PHONY: html pdf clean

html: $(HTML)

pdf: $(PDF)

clean:
	-rm -f $(HTML) $(PDF)

%.html: %.md
	$(FACTORY) $(OPTIONS) $<

%.pdf: %.md
	$(FACTORY) $(OPTIONS) --pdf $<
