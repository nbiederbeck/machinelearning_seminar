all: build/kickoff.pdf

build:
	mkdir -p build

build/kickoff.pdf: FORCE | build
	latexmk \
		--lualatex \
		--interaction=nonstopmode \
		--halt-on-error \
		--output-directory=build \
	kickoff.tex

preview: kickoff.tex | build
	latexmk \
		--lualatex \
		--interaction=nonstopmode \
		--halt-on-error \
		--output-directory=build \
	kickoff.tex --pvc

FORCE:

clean:
	rm -rf build

.PHONY: all clean preview FORCE
