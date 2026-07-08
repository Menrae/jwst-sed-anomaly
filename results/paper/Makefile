MAIN := main
TEX  := pdflatex -interaction=nonstopmode -halt-on-error
BIB  := bibtex

ARXIV_NAME := $(MAIN)-arxiv
ARXIV_FILES := $(MAIN).tex references.bib $(wildcard *.bst) $(wildcard *.cls) \
               $(wildcard *.pdf) $(wildcard *.png) $(wildcard *.eps)

.PHONY: pdf clean arxiv

pdf:
	$(TEX) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz \
	      $(MAIN).lof $(MAIN).lot $(MAIN).toc
	rm -f $(ARXIV_NAME).zip

arxiv: clean pdf
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).fls $(MAIN).fdb_latexmk
	rm -f $(ARXIV_NAME).zip
	if command -v zip >/dev/null 2>&1; then \
		zip -j $(ARXIV_NAME).zip $(ARXIV_FILES) $(MAIN).bbl; \
	else \
		python3 -c "import zipfile, sys, os; zf = zipfile.ZipFile(sys.argv[1], 'w'); [zf.write(p, os.path.basename(p)) for p in sys.argv[2:]]; zf.close()" \
			$(ARXIV_NAME).zip $(ARXIV_FILES) $(MAIN).bbl; \
	fi
