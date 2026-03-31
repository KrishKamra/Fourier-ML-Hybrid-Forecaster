import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Evaluator:
    @staticmethod
    def get_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

if __name__ == "__main__":
    import numpy as np
    # Mock data: Actual vs Predicted
    actual = np.array([10.0, 12.0, 15.0, 11.0])
    predicted = np.array([10.2, 11.8, 15.5, 10.9])
    
    results = Evaluator.get_metrics(actual, predicted)
    print("--- Model Evaluation Metrics ---")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")