FACTORY=slidefactory.sif
OPTIONS=

SRC  = $(wildcard *.md)
HTML = $(patsubst %.md,%.html,$(SRC))
PDF  = $(patsubst %.md,%.pdf,$(SRC))
DOCS = $(wildcard ../*/docs)

.PHONY: all html pdf clean $(DOCS)

all: $(DOCS) $(PDF)

html: $(HTML)

pdf: $(PDF)

clean:
	-rm -f $(HTML) $(PDF)

$(DOCS):
	@cd $@ && echo $@ && $(MAKE) --no-print-directory pdf

%.html: %.md
	$(FACTORY) $(OPTIONS) $<

%.pdf: %.md
	$(FACTORY) $(OPTIONS) --pdf $<
