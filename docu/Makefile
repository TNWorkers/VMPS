FILES = $(wildcard *.tex)
PDFS = $(subst .tex,.pdf,$(FILES))

%.pdf: %.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape --interaction=nonstopmode" $*.tex #$(.SOURCE)

all: $(PDFS) 

PHONY:

clean:
	latexmk -CA
	rm -f ./svg/*.pdf_tex
	rm -f ./svg/*.pdf
