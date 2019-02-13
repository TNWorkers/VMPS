FILES = $(wildcard *.tex)
FILES_DOXY = structure_factor.tex #cgc.tex

XHTMLS = $(patsubst %.tex,../extern-html/%.xhtml,$(FILES_DOXY))
$(info $$XHTMLS is [${XHTMLS}])

PDFS = $(subst .tex,.pdf,$(FILES))

%.pdf: %.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape --interaction=nonstopmode" $*.tex

../extern-html/%.xhtml: %.tex
	latexml -dest ../extern-html/$*.xml $*.tex
	latexmlpost -dest ../extern-html/$*.xhtml ../extern-html/$*.xml
	rm ../extern-html/$*.xml

all: $(PDFS) 

doxy: $(XHTMLS)

PHONY:

clean:
	latexmk -CA
	rm -f ./svg/*.pdf_tex
	rm -f ./svg/*.pdf
