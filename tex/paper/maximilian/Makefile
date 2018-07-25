BUILD_FILES = build/main.pdf


OUT_DIR = build
TEX_CMD = latexmk \
	--pdf \
	--output-directory=$(OUT_DIR) \
	--halt-on-error \
	--interaction=nonstopmode
PREVIEW_CMD = \
	$(TEX_CMD) --pvc


all: $(BUILD_FILES)


build/main.pdf: FORCE | build
	TEXINPUTS=$(OUT_DIR): BIBINPUTS=$(OUT_DIR): max_print_line=1048576 \
	$(TEX_CMD) main.tex


preview: FORCE | build
	TEXINPUTS=$(OUT_DIR): BIBINPUTS=$(OUT_DIR): max_print_line=1048576 \
	$(PREVIEW_CMD) main.tex


FORCE:


build:
	@mkdir -p build


clean:
	rm -rf *.pdf build

.PHONY: all clean FORCE preview
