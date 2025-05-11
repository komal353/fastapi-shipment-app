from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline
pipeline = joblib.load("delivery_pipeline.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input schema (same as your raw dataset excluding ID and target)
class ShipmentFeatures(BaseModel):
    Warehouse_block: str
    Mode_of_Shipment: str
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: int
    Prior_purchases: int
    Product_importance: str
    Gender: str
    Discount_offered: int
    Weight_in_gms: int

@app.post("/predict")
def predict(data: ShipmentFeatures):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Ensure all input features are of correct type and format
        if input_df.isnull().values.any():
            raise ValueError("Missing values detected in input data.")
        
        # Predict using the pipeline (handles encoding, scaling, etc.)
        prediction = pipeline.predict(input_df)[0]

        # Return the prediction in a dictionary format
        return {"prediction": int(prediction)}

    except Exception as e:
        # Catch all unexpected errors and return a 400 error response
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")