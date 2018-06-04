all: build/kickoff.pdf

build:
	mkdir -p build

build/kickoff.pdf: kickoff.tex | build
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
