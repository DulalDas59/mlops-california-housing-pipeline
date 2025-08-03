# api/schema.py

from pydantic import BaseModel
from typing import List

class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class BatchRequest(BaseModel):
    instances: List[Features]
