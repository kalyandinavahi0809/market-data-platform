.PHONY: install lint test

install:
	pip install -r requirements.txt

lint:
	black src

test:
	pytest
