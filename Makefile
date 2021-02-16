PYTHON = python3

.PHONY = train test predict all

.DEFAULT_GOAL = all

all:
	${PYTHON} train.py
	${PYTHON} test.py

train:
	${PYTHON} train.py

test:
	${PYTHON} test.py

predict:
	${PYTHON} predict.py
