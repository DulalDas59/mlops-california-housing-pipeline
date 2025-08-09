from typing import List  # or: from typing import Annotated
from pydantic import BaseModel, Field, model_validator

class Features(BaseModel):
    MedInc: float = Field(..., ge=0, le=20, description="Median income in tens of thousands")
    HouseAge: float = Field(..., ge=0, le=100)
    AveRooms: float = Field(..., gt=0, le=50)
    AveBedrms: float = Field(..., gt=0, le=10)
    Population: float = Field(..., ge=1, le=100000)
    AveOccup: float = Field(..., ge=0.1, le=20)
    Latitude: float = Field(..., ge=32.0, le=42.0)
    Longitude: float = Field(..., ge=-125.0, le=-114.0)

    # Cross-field checks in v2
    @model_validator(mode="after")
    def check_rooms_vs_bedrooms(self):
        if self.AveRooms <= self.AveBedrms:
            raise ValueError("AveRooms must be greater than AveBedrms")
        return self

    # Optional: forbid unknown fields
    model_config = {"extra": "forbid"}

class BatchRequest(BaseModel):
    instances: List[Features]  # or: list[Features] in Python 3.9+
    model_config = {"extra": "forbid"}
