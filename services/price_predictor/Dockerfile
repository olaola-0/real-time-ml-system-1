# Base image with Python
FROM python:3.10.3-slim-buster

# Install gcc and python3-dev to compile Python packages
RUN apt-get update && apt-get install -y \
    gcc wget build-essential \
    python3-dev libssl-dev libffi-dev \
    ca-certificates

# Install additional libraries needed for scipy
RUN apt-get install -y gfortran libopenblas-dev liblapack-dev

# Download and install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# stream output to console
ENV PYTHONUNBUFFERED=1

# Install poetry inside the container
RUN pip install poetry==1.8.2

# Configure poetry to create virtualenvs inside the project directory
RUN poetry config virtualenvs.create false

WORKDIR /app

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock /app/

# Copy the tools directory into the container
COPY tools /app/tools

# Install Python dependencies from the pyproject.toml file
RUN poetry install --no-root

# Copy all the source code into the container
COPY . /app

# Expose the port that the application will run on
EXPOSE 80

# Command to run the application with Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:80", "src.api:app"]