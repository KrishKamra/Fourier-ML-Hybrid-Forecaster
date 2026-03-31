import numpy as np
import pandas as pd

class DataFetcher:
    def get_synthetic_data(self, n_points=500):
        """Generates a signal: Baseline + 24h Cycle + 168h Cycle + Noise"""
        np.random.seed(42) # For reproducibility
        time = np.arange(n_points)
        
        # Daily cycle (sine wave with period of 24)
        daily = 10 * np.sin(2 * np.pi * time / 24)
        # Weekly cycle (sine wave with period of 168)
        weekly = 5 * np.sin(2 * np.pi * time / 168)
        # Random Noise
        noise = np.random.normal(0, 2, n_points)
        
        series = daily + weekly + noise + 20 # +20 is the DC offset (average)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2026-01-01', periods=n_points, freq='h'),
            'value': series
        })
        return df