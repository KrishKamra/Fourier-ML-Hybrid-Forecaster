from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from feature_engineering.fourier_features import FourierFeatureExtractor
from models.xgboost_model import XGBoostForecaster
from data_ingestion.api_fetcher import DataFetcher
from preprocessing.window_generator import WindowGenerator

app = FastAPI(title="Fourier-Enhanced AI Forecaster")

# --- Global State (In a real app, you'd load a saved model file) ---
# For now, we train a quick model on startup so the API is functional
fetcher = DataFetcher()
extractor = FourierFeatureExtractor(top_k=2)
window_gen = WindowGenerator(input_width=48)
forecaster = XGBoostForecaster()

@app.on_event("startup")
def train_model():
    print("API Booting: Training core model...")
    df = fetcher.get_synthetic_data(n_points=1000)
    series = df['value'].values
    X_raw, y = window_gen.split_windows(series)
    
    enhanced = [list(w) + extractor.extract_feature_vector(w) for w in X_raw]
    forecaster.train(np.array(enhanced), y.flatten())
    print("API Ready!")

# --- Data Models ---
class ForecastRequest(BaseModel):
    data_window: list  # Expecting a list of 48 floats

# --- Routes ---
@app.post("/predict")
async def predict(request: ForecastRequest):
    if len(request.data_window) != 48:
        raise HTTPException(status_code=400, detail="Window must be exactly 48 points.")
    
    # 1. Extract Fourier DNA from the incoming request
    window = np.array(request.data_window)
    f_vector = extractor.extract_feature_vector(window)
    
    # 2. Combine Raw + Fourier
    full_features = np.array(list(window) + f_vector).reshape(1, -1)
    
    # 3. Inference
    prediction = forecaster.predict(full_features)
    
    return {
        "prediction": float(prediction[0]),
        "detected_cycles": extractor.get_dominant_features(window)["periods"]
    }