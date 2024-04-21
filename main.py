from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib


app = FastAPI(
    title="Deploy Social Network Adds",
    version="0.0.1"
)

# ------------------------------------------------------------
# LOAD THE AI MODEL
# ------------------------------------------------------------
model = joblib.load("model/linear_regression_model_v01.pkl")


@app.post("/predict/")
async def predict(data: dict):
    try:
        # Convertir los datos de entrada en un DataFrame
        input_data = pd.DataFrame([data])

        # Realizar la predicción utilizando el modelo cargado
        prediction = model.predict(input_data)

        # Devolver la predicción como JSON utilizando JSONResponse
        return JSONResponse(content={"prediction": prediction[0]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
