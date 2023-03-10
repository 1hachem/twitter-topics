format:
	black .
	isort .

test:
	pytest ./tests/test.py

