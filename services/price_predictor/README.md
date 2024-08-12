# Price Predictor Service

The Price Predictor Service is a machine learning application designed to predict the price of financial products. It is built using Python and is containerized using Docker for easy deployment and scaling.

## Features

The service consists of several components:

- **Training Pipeline**: Trains a predictive model using historical data.
- **Inference Pipeline**: Uses the trained model to make predictions on live data.
- **API**: A RESTful API that allows other applications to use the predictive model.

## Usage

The service is controlled using a Makefile, which automates various tasks related to the training, prediction, and deployment of the model.

### Train the model

```makefile
make train
```

This command sets up necessary environment variables and runs the training script.

### Run predictions

```makefile
make predict
```

This command sets up necessary environment variables and runs the prediction script.

### Start the API

```makefile
make api
```

This command sets up necessary environment variables and starts the Flask API server.

### Send a request to the API

```makefile
make request
```

This command sends a POST request to the Flask API server with a JSON payload containing a `product_id`.

### Send an invalid request to the API

```makefile
make invalid-request
```

This command sends a POST request to the Flask API server with a JSON payload containing an invalid `product_id`.

### Copy tools

```makefile
make copy-tools
```

This command copies a directory named `tools` from a parent directory into the current directory.

### Build the Docker image

```makefile
make build
```

This command builds a Docker image of the application.

### Run the Docker container

```makefile
make run
```

This command runs the Docker container, exposing it on port 5005.

### Send a request to the production API

```makefile
make request-production
```

This command is currently not implemented.

### Lint the code

```makefile
make lint
```

This command checks the code for stylistic errors and automatically fixes them.

### Format the code

```makefile
make format
```

This command formats the code according to a predefined style.

### Lint and format the code

```makefile
make lint-and-format
```

This command lints and formats the code.

## Tools Used

- **Python**: The main programming language used for implementing the service.
- **Poetry**: A tool for dependency management and packaging in Python.
- **Flask**: A lightweight web framework used for creating the API.
- **Docker**: A platform used to containerize the application.
- **Curl**: A command-line tool used for sending HTTP requests.
- **Makefile**: A file containing a set of directives used by a make build automation tool to generate a target/goal.