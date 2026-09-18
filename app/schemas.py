from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    total_price: float = Field(..., gt=0, description="Price must be greater than 0")
    total_freight: float = Field(
        ..., ge=0, description="Freight cost cannot be negative"
    )
    total_items: int = Field(
        ..., gt=0, description="Number of items must be at least 1"
    )
    total_payment: float = Field(
        ..., gt=0, description="Payment amount must be greater than 0"
    )
    max_installments: int = Field(
        ..., ge=1, description="Number of installments must be at least 1"
    )

    order_status: str
    order_approved_at: str
    order_delivered_carrier_date: str

    customer_state: str = Field(
        ...,
        min_length=2,
        max_length=2,
        pattern="^[A-Z]{2}$",
        description="State code must be exactly two uppercase English letters",
    )


class PredictionOutput(BaseModel):
    prediction: float = Field(..., description="Prediction value output by the model")
