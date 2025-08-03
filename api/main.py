# api/main.py

from fastapi import FastAPI
from api.schema import BatchRequest
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/dt.pkl")

@app.get("/")
def root():
    return {"message": "California Housing Model is Live 🚀"}

@app.post("/predict")
def predict(data: BatchRequest):
    X = [[
        d.MedInc, d.HouseAge, d.AveRooms, d.AveBedrms,
        d.Population, d.AveOccup, d.Latitude, d.Longitude
    ] for d in data.instances]

    preds = model.predict(X)
    return {"predictions": preds.tolist()}
