.PHONY: help install install-dev setup train train-simple eval test clean api docker-build docker-run lint format

help:
	@echo "CIFAR-10 CNN Classifier - Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make setup         - Create project structure"
	@echo "  make train         - Train the model (enhanced architecture)"
	@echo "  make train-simple  - Train the model (simple architecture)"
	@echo "  make eval          - Evaluate trained model"
	@echo "  make test          - Run tests"
	@echo "  make api           - Start FastAPI server"
	@echo "  make clean         - Clean generated files"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

setup:
	python create_structure.py

train:
	python -m source.train --epochs 10 --batch-size 64

train-simple:
	python -m source.train --epochs 10 --batch-size 64 --simple

eval:
	python -m source.train --eval-only

test:
	python -m pytest tests/ -v --cov=source --cov-report=html

test-simple:
	python -m unittest discover -s tests -v

api:
	uvicorn source.api.app:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	uvicorn source.api.app:app --host 0.0.0.0 --port 8000 --workers 4

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-data:
	rm -rf data/*
	rm -rf checkpoints/*
	rm -rf sample/plots/*
	rm -rf logs/*

lint:
	flake8 source tests --max-line-length=100 --ignore=E501,W503

format:
	black source tests --line-length=100

docker-build:
	docker build -t cifar10-classifier .

docker-run:
	docker run -p 8000:8000 cifar10-classifier

notebook:
	jupyter notebook