.PHONY: help setup train-all build up down restart logs test loadtest clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install Python dependencies locally
	pip install -r training/requirements.txt
	pip install -r gateway/requirements.txt
	pip install -r tests/requirements.txt

generate-data: ## Generate synthetic fraud dataset
	python data/generate_dataset.py

train-xgboost: ## Train XGBoost model
	python training/train_xgboost.py

train-pytorch: ## Train PyTorch model
	python training/train_pytorch.py

train-all: generate-data train-xgboost train-pytorch ## Generate data and train all models

build: ## Build all Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose down && docker-compose up -d

logs: ## Tail logs for all services
	docker-compose logs -f

test: ## Run unit tests
	python -m pytest tests/ -v --tb=short

loadtest: ## Run Locust load test (30s, 50 users)
	cd loadtest && locust -f locustfile.py --headless -u 50 -r 10 -t 30s --host http://localhost:8000

clean: ## Remove generated data and models
	rm -f data/transactions.csv
	rm -f models/*.json models/*.pt
	rm -rf mlruns/
