SHELL := /bin/bash

train:
	source ./setup_credentials.sh && \
	poetry run python src/training.py

predict:
	source ./setup_credentials.sh && \
	poetry run python src/predictor.py

api:
	source ./setup_credentials.sh && \
	poetry run python src/api.py

request:
	curl -X POST http://127.0.0.1:5005/predict \
	-H "Content-Type: application/json" \
	-d '{"product_id":"BTC/USD"}'

invalid-request:
	curl -X POST http://127.0.0.1:5005/predict \
	-H "Content-Type: application/json" \
	-d '{"product_id":"ETH/USD"}'
	
copy-tools:
	cp -r ../../tools .

build: copy-tools
	docker build -t price_predictor-api .

run: build
	docker run \
		--env-file ./setup_credentials.sh \
		-p 5005:80 \
		price_predictor-api

request-production:
	

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .

lint-and-format: lint format