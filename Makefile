install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest

run:
	uvicorn app:app --reload

