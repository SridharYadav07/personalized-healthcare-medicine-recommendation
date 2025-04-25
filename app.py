from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("medication_model.joblib")
med_encoder = joblib.load("medication_label_encoder.joblib")


encoders = {
    'Gender': joblib.load("Gender_encoder.joblib"),
    'Blood Type': joblib.load("Blood Type_encoder.joblib"),
    'Medical Condition': joblib.load("Medical Condition_encoder.joblib"),
    'Insurance Provider': joblib.load("Insurance Provider_encoder.joblib"),
    'Admission Type': joblib.load("Admission Type_encoder.joblib"),
    'Test Results': joblib.load("Test Results_encoder.joblib")
}


app = FastAPI()

class Patient(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Insurance_Provider: str
    Billing_Amount: float
    Admission_Type: str
    Length_of_Stay: int
    Test_Results: int

@app.get("/")
def root():
    return{"message": "Healthcare Recommendation API is running"}

@app.post("/predict/")
def predict_medication(patient: Patient):
    try:
        gender = encoders['Gender'].transform([patient.Gender])[0]
        blood_type = encoders['Blood Type'].transform([patient.Blood_Type])[0]
        condition = encoders['Medical Condition'].transform([patient.Medical_Condition])[0]
        insurance = encoders['Insurance Provider'].transform([patient.Insurance_Provider])[0]
        adm_type = encoders['Admission Type'].transform([patient.Admission_Type])[0]
        test_result = encoders['Test Results'].transform([patient.Test_Results])[0]



        input_data = np.array([
            patient.Age,
            gender,
            blood_type,
            condition,
            insurance,
            patient.Billing_Amount,
            adm_type,
            patient.Length_of_Stay,
            test_result
        ]).reshape(1, -1)

        pred = model.predict(input_data)[0]
        med_name = med_encoder.inverse_transform([pred])[0]

        return {"recommended_medication": med_name}
    
    except Exception as e:
        return {"error": str(e)}