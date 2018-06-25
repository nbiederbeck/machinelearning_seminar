gifs = $(addprefix build/, $(addsuffix .gif, \
	rg gb br \
))

all: $(gifs)

build/rg.gif: build/rg_009.png
	convert -delay 0 -loop 0 build/rg_*.png build/rg.gif

build/gb.gif: build/gb_009.png
	convert -delay 0 -loop 0 build/gb_*.png build/gb.gif

build/br.gif: build/br_009.png
	convert -delay 0 -loop 0 build/br_*.png build/br.gif

$(addprefix build/, $(addsuffix _009.png, rg gb br)): generate_color_gifs.py
	python $<
