DO = \
	 slides/kickoff

all:
	for d in $(DO); do \
		cd $$d && make ; \
	done

clean:
	for d in $(DO); do \
		cd $$d && make clean ; \
	done
