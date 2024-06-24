from pydantic import BaseModel


class Trade(BaseModel):
    """A class to represent a trade using Pydantic."""

    product_id: str
    price: float
    volume: float
    timestamp_ms: int
