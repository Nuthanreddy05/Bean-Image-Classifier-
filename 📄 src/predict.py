import numpy as np
import joblib

# Load trained pipeline
pipeline = joblib.load('model_pipeline.pkl')

def predict(input_df):
    X_scaled = pipeline['scaler'].transform(input_df)
    preds = pipeline['model'].predict(X_scaled)
    labels = pipeline['encoder'].inverse_transform(preds)
    return labels
