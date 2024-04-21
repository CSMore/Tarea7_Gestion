from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
# import joblib

app = FastAPI(
    title="Deploy Social Network Adds",
    version="0.0.1"
)


# ------------------------------------------------------------
# LOAD THE AI MODEL
# ------------------------------------------------------------
# model = joblib.load("model/linear_regression_model_v01.pkl")


@app.post(path="/api/v1/predict-social-network-adds", tags=["social-network-adds"])
async def predict(
        gender: float,
        age: float,
        estimated_salary: float
):
    dictionary = {
        "Gender": gender,
        "Age": age,
        "EstimatedSalary": estimated_salary
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        # prediction = model.predict(df)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=1
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )
