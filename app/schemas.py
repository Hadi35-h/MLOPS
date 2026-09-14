from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    total_price: float = Field(..., gt=0, description="يجب أن يكون السعر أكبر من 0")
    total_freight: float = Field(
        ..., ge=0, description="تكلفة الشحن لا يمكن أن تكون سالبة"
    )
    total_items: int = Field(
        ..., gt=0, description="عدد العناصر يجب أن يكون 1 على الأقل"
    )
    total_payment: float = Field(
        ..., gt=0, description="المبلغ المدفوع يجب أن يكون أكبر من 0"
    )
    max_installments: int = Field(
        ..., ge=1, description="عدد الأقساط يجب أن يكون 1 على الأقل"
    )

    order_status: str
    order_approved_at: str
    order_delivered_carrier_date: str

    customer_state: str = Field(
        ...,
        min_length=2,
        max_length=2,
        pattern="^[A-Z]{2}$",
        description="رمز الولاية يجب أن يتكون من حرفين كبيرين بالإنجليزية فقط",
    )


class PredictionOutput(BaseModel):
    prediction: float = Field(..., description="قيمة التوقع الناتجة من النموذج")
