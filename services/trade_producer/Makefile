SHELL := /bin/bash

run-dev:
	@echo "Running development environment with live data"
	KAFKA_BROKER_ADDRESS='localhost:19092' \
	source ./setup_live_config.sh && poetry run python src/main.py

run-dev-historical:
	@echo "Running development environment with historical data"
	KAFKA_BROKER_ADDRESS='localhost:19092' \
	source ./setup_historical_config.sh && poetry run python src/main.py

build:
	docker build -t trade-producer .

run: build
	docker run \
		--network=redpanda_network \
		-e KAFKA_BROKER_ADDRESS='redpanda-0:9092' \
		-e KAFKA_TOPIC=trade \
		-e PRODUCT_IDS='["BTC/USD"]' \
		-e LIVE_OR_HISTORICAL=live \
		trade-producer

run-historical: build
	docker volume create trade-producer-volume
	docker run \
		--network=redpanda_network \
		-e KAFKA_BROKER_ADDRESS='redpanda-0:9092' \
		-e KAFKA_TOPIC=trade_historical \
		-e PRODUCT_IDS='["BTC/USD"]' \
		-e LIVE_OR_HISTORICAL=historical \
		-e LAST_N_DAYS=90 \
		-e CACHE_DIR_HISTORICAL_DATA=/tmp/historical_trade_data \
		-v trade-producer-volume:/tmp/historical_trade_data \
		trade-producer
		

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .

lint-and-format: lint format