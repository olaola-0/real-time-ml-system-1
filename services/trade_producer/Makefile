build:
	docker build -t trade_producer .

run: build
	docker run -it --network=redpanda_network trade_producer

lint:
	ruff check --fix

format:
	ruff format .