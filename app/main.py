from fastapi import FastAPI

from app.schemas import EmployeeData
from app.predictor import predict_employee


app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predicts whether an employee is likely to leave the company using a machine learning model trained on the IBM HR Analytics Employee Attrition dataset.",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "Employee Attrition Prediction API"
    }


@app.post("/predict")
def predict(data: EmployeeData):

    result = predict_employee(
        data.model_dump()
    )

    return result
