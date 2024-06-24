SHELL := /bin/bash

run-dev:
	@echo "Running trade-to-ohlc in development mode"
	KAFKA_BROKER_ADDRESS='localhost:19092' \
	source ./setup_live_config.sh && poetry run python src/main.py

run-dev-historical:
	@echo "Running trade-to-ohlc in development mode with historical data"
	KAFKA_BROKER_ADDRESS='localhost:19092' \
	source ./setup_historical_config.sh && poetry run python src/main.py

build:
	docker build -t trade-to-ohlc .

run: build
	docker run \
		--network=redpanda_network \
		--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
		--env KAFKA_INPUT_TOPIC=trade \
		--env KAFKA_OUTPUT_TOPIC=ohlc \
		--env KAFKA_CONSUMER_GROUP=trade_to_ohlc_consumer_group \
		--env OHLC_WINDOW_SECONDS=60 \
		trade-to-ohlc

run-historical: build
	docker run \
		--network=redpanda_network \
		--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
		--env KAFKA_INPUT_TOPIC=trade_historical \
		--env KAFKA_OUTPUT_TOPIC=ohlc_historical \
		--env KAFKA_CONSUMER_GROUP=trade_to_ohlc_historical_consumer_group \
		--env OHLC_WINDOW_SECONDS=60 \
		trade-to-ohlc

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .

lint-and-format: lint format