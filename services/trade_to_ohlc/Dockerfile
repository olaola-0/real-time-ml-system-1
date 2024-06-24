# Builder stage:
# Dockerfile
# Builder stage: This stage is used to install the dependencies and create a virtual environment
FROM python:3.10.3-slim-buster as builder

# Install poetry inside the container and set the version to the local machine version
# This is done to avoid any compatibility issues
RUN pip install poetry==1.7.1

# Set environment variables for poetry 
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the container
COPY pyproject.toml poetry.lock ./

# Install dependencies with BuildKit cache
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

# Runtime stage: Use a smaller, more secure image for the runtime environment
FROM python:3.10.3-slim-buster as runtime

# Set environment variables for the virtual environment
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="${PYTHONPATH}:/app/src"

# Copy the virtual environment from the builder stage
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy the application source code
COPY src ./src

# Set the default command to run the application
CMD ["python", "src/main.py"]
