BUILD_DIR = ./build
DIST_DIR = ./dist

.PHONY: all

all: $(BUILD_DIR)/exportlist.bib $(DIST_DIR)/main.pdf

$(BUILD_DIR)/exportlist.bib: exportlist.bib
	mkdir -p $(BUILD_DIR)
	cp exportlist.bib $(BUILD_DIR)/


$(DIST_DIR)/main.pdf: main.tex $(BUILD_DIR)/exportlist.bib
	mkdir -p $(BUILD_DIR)
	mkdir -p $(DIST_DIR)
	pdflatex -output-directory=$(BUILD_DIR) main.tex
	pdflatex -output-directory=$(BUILD_DIR) main.tex
	cp $(BUILD_DIR)/main.pdf $(DIST_DIR)/main.pdf
