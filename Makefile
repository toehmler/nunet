PYTHON = python3

.PHONY = train 

.DEFAULT_GOAL = train

train:
	${PYTHON} train.py



