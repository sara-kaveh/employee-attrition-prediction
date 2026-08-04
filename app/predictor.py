import joblib
import pandas as pd

artifact = joblib.load("models/employee_attrition_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]


def predict_employee(employee: dict):

    df = pd.DataFrame([employee])

    probability = model.predict_proba(df)[0, 1]

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": threshold
    }
