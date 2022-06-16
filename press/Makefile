FACTORY=slidefactory.sif
OPTIONS=--config width=905 --config height=1280 -t csc-2018-portrait

SRC  = $(wildcard *.md)
HTML = $(patsubst %.md,%.html,$(SRC))
PDF  = $(patsubst %.md,%.pdf,$(SRC))
DOCS = $(wildcard ../*/docs)

HANDOUT=lecture-slides.pdf

.PHONY: all html pdf jam clean $(DOCS)

all: $(DOCS) $(PDF)

html: $(HTML)

pdf: $(PDF)

jam:
	./jam-it.sh

clean:
	-rm -f $(HTML) $(PDF) $(HANDOUT)

$(DOCS):
	@cd $@ && echo $@ && $(MAKE) --no-print-directory pdf

%.html: %.md
	$(FACTORY) $(OPTIONS) $<

%.pdf: %.md
	$(FACTORY) $(OPTIONS) --pdf $<
