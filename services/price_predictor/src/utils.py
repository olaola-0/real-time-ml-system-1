def get_model_name(product_id: str) -> str:
    """
    Returns the model registry name for the given product_id

    Args:
        product_id (str): Product ID of the model to be fetched

    Returns:
        str: Model registry name
    """
    return f"{product_id.replace('/', '_')}_price_change_predictor"