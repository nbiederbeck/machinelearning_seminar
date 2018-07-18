all: build/final.pdf

build:
	mkdir -p build

build/final.pdf: FORCE | build
	lualatex \
		--interaction=nonstopmode \
		--halt-on-error \
		--output-directory=build \
	final.tex

FORCE:

clean:
	rm -rf build

.PHONY: all clean preview FORCE
