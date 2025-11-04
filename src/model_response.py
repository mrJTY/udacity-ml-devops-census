from pydantic import BaseModel, Field
from typing import Literal


class ModelResponse(BaseModel):
    """
    Response schema for census income prediction.

    Returns the predicted salary category and prediction probability.
    """
    predicted_salary: Literal["<=50K", ">50K"] = Field(
        ...,
        description="Predicted salary category: <=50K or >50K"
    )
    prediction_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the predicted class (confidence score)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_salary": ">50K",
                "prediction_prob": 0.87
            }
        }