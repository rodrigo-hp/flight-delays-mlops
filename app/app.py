import pandas as pd
import joblib

from pydantic import BaseModel
from fastapi import FastAPI


class FlightInformation(BaseModel):
    DIA: int
    MES: int
    DIANOM: str
    TIPOVUELO: str
    OPERA: str
    SIGLADES: str
    TEMPORADAALTA: int
    PERIODODIA: str


class PredictionOut(BaseModel):
    delay_proba: float


# Load the model
model_data = {}
model_data = joblib.load("/app/flight_delays_lgb_model.pkl")
model = model_data["trained_model"]

# Start the app
app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "Flight Delays API", "model_version": 0.1}


# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: FlightInformation):
    request_df = pd.DataFrame([payload.dict()])
    request_df["DIA"] = pd.Categorical(
        request_df["DIA"], categories=model_data["dia_values"]
    )
    request_df["MES"] = pd.Categorical(
        request_df["MES"], categories=model_data["mes_values"]
    )
    request_df["DIANOM"] = pd.Categorical(
        request_df["DIANOM"], categories=model_data["dianom_values"]
    )
    request_df["TIPOVUELO"] = pd.Categorical(
        request_df["TIPOVUELO"], categories=model_data["tipovuelo_values"]
    )
    request_df["OPERA"] = pd.Categorical(
        request_df["OPERA"], categories=model_data["opera_values"]
    )
    request_df["SIGLADES"] = pd.Categorical(
        request_df["SIGLADES"], categories=model_data["siglades_values"]
    )
    request_df["PERIODODIA"] = pd.Categorical(
        request_df["PERIODODIA"], categories=model_data["periododia_values"]
    )
    prediction = model.predict_proba(request_df)[0, 1]
    result = {"delay_proba": prediction}
    return result
