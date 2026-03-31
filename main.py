from data_ingestion.api_fetcher import DataFetcher
from feature_engineering.fourier_features import FourierFeatureExtractor
from preprocessing.window_generator import WindowGenerator
from models.xgboost_model import XGBoostForecaster
import pandas as pd 
import numpy as np

def run_diagnostic():
    print("--- Initializing Phase 1: Signal Diagnostic ---")
    
    # Initialize components
    fetcher = DataFetcher()
    extractor = FourierFeatureExtractor(top_k=2)
    
    # 1. Fetch data
    df = fetcher.get_synthetic_data()
    print(f"Data Loaded: {len(df)} rows.")
    
    # 2. Extract features from a 1-week window (168 hours)
    window = df['value'].iloc[:168].values
    features = extractor.get_dominant_features(window)
    
    # 3. Display Findings
    print("\nDominant Cycles Detected:")
    for i, period in enumerate(reversed(features['periods'])):
        print(f"  Cycle {i+1}: ~{round(period, 2)} hours")


def run_phase_2():
    print("--- Phase 2: Feature Engineering & Windowing ---")
    
    # 1. Setup
    fetcher = DataFetcher()
    extractor = FourierFeatureExtractor(top_k=2)
    window_gen = WindowGenerator(input_width=48, label_width=1)
    
    # 2. Get Raw Data
    df = fetcher.get_synthetic_data(n_points=200) # Smaller set for demo
    series = df['value'].values
    
    # 3. Create Windows
    X_raw, y = window_gen.split_windows(series)
    
    # 4. "Enhance" each window with Fourier Features
    enhanced_features = []
    for window in X_raw:
        f_vector = extractor.extract_feature_vector(window)
        # Combine [Raw 48 hours] + [4 Fourier Features]
        combined = list(window) + f_vector
        enhanced_features.append(combined)
    
    # 5. Review the shape
    X_final = pd.DataFrame(enhanced_features)
    print(f"Original Window Size: {X_raw.shape[1]}")
    print(f"Enhanced Feature Size: {X_final.shape[1]} (48 raw + 4 Fourier)")
    print(f"Total Training Samples: {len(X_final)}")
    
    print("\nSample of Enhanced Row (Last 5 columns):")
    print(X_final.iloc[0, -5:]) 

def run_phase_3():
    print("--- Phase 3: Model Training ---")
    
    # 1. Setup
    fetcher = DataFetcher()
    extractor = FourierFeatureExtractor(top_k=2)
    window_gen = WindowGenerator(input_width=48, label_width=1)
    
    # 2. Get More Data (Need more for training)
    df = fetcher.get_synthetic_data(n_points=1000)
    series = df['value'].values
    
    # 3. Create Windows & Features
    X_raw, y = window_gen.split_windows(series)
    enhanced_features = []
    for window in X_raw:
        f_vector = extractor.extract_feature_vector(window)
        enhanced_features.append(list(window) + f_vector)
    
    X = np.array(enhanced_features)
    y = y.flatten() # XGBoost expects a 1D array for labels
    
    # 4. Train Model
    forecaster = XGBoostForecaster()
    forecaster.train(X, y)
    
    # 5. Run a Test Prediction
    sample_input = X[-1].reshape(1, -1)
    prediction = forecaster.predict(sample_input)
    actual = y[-1]
    
    print(f"\nFinal Test Prediction:")
    print(f"  Predicted: {prediction[0]:.2f}")
    print(f"  Actual:    {actual:.2f}")
    print(f"  Difference: {abs(prediction[0] - actual):.2f}")

if __name__ == "__main__":
    # run_diagnostic() # Commented out Phase 1
    # run_phase_2()    # Commented out Phase 2
    run_phase_3()
