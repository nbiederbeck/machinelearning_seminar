all: build/final.pdf

build:
	mkdir -p build

build/final.pdf: FORCE content/hist.pdf | build
	lualatex \
		--interaction=nonstopmode \
		--halt-on-error \
		--output-directory=build \
	final.tex

content/hist.pdf: hist.py
	python hist.py

FORCE:

clean:
	rm -rf build

.PHONY: all clean preview FORCE
