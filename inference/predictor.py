import numpy as np

class Predictor:
    def __init__(self, model, extractor):
        self.model = model
        self.extractor = extractor

    def forecast(self, window):
        f_vector = self.extractor.extract_feature_vector(window)
        combined = np.array(list(window) + f_vector).reshape(1, -1)
        return self.model.predict(combined)[0]
    
if __name__ == "__main__":
    from data_ingestion.api_fetcher import DataFetcher
    from feature_engineering.fourier_features import FourierFeatureExtractor
    from xgboost import XGBRegressor # Assuming model type
    
    # Mock setup
    fetcher = DataFetcher()
    extractor = FourierFeatureExtractor(top_k=3)
    mock_model = XGBRegressor().fit(np.random.rand(10, 54), np.random.rand(10))
    
    predictor = Predictor(mock_model, extractor)
    sample_window = fetcher.get_synthetic_data(n_points=48)['value'].values
    
    prediction = predictor.forecast(sample_window)
    print(f"--- Inference Test ---")
    print(f"Input Window (last 5 pts): {sample_window[-5:]}")
    print(f"Future Forecast: {prediction:.4f}")