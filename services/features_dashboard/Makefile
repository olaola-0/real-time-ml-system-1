SHELL := /bin/bash

run-dev:
	source ./setup_credentials.sh && \
	source ./setup_config.sh && \
	poetry run streamlit run src/frontend.py --server.runOnSave true

fetch-offline-data:
	source ./setup_credentials.sh && \
	source ./setup_config.sh && \
	poetry run python src/backend.py --offline

fetch-online-data:
	source ./setup_credentials.sh && \
	source ./setup_config.sh && \
	poetry run python src/backend.py --online

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .

lint-and-format: lint format