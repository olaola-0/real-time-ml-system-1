from flask import Flask, jsonify, request

from src.predictor import Predictor

# List of cryptocurrencies supported by the predictor
SUPPORTED_PRODUCT_IDS = ['BTC/USD']

# Initialize the Flask app
app = Flask(__name__)

# Initialize the predictor
# NB: If we want want to serve preditions for multiple products, we can load the predictor for each product
# and store them in a dictionary with the product_id as the key.
predictors = {
    product_id: Predictor.from_model_registry(
        product_id=product_id, status='production'
    )
    for product_id in SUPPORTED_PRODUCT_IDS
}


@app.route('/health')
def health():
    return 'I am Healthy!'


# Define the endpoint for the price prediction service, post method
@app.route('/predict', methods=['POST'])
def predict():
    """
    Generates a price prediction for a given product_id using the predictor object and returns it as a JSON object
    """
    # Get the product_id from the request
    product_id = request.json.get('product_id')

    # Check if the product_id is supported
    if product_id not in SUPPORTED_PRODUCT_IDS:
        return jsonify({'error': f'Product ID {product_id} is not supported'}), 400

    # Get the predictor object for the product_id
    predictor = predictors[product_id]

    output = predictor.predict()

    return jsonify(output.to_dict())


if __name__ == '__main__':
    app.run(port=5000, debug=True)