import xgboost as xgb
from sklearn.model_selection import train_test_split

class XGBoostForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror'
        )

    def train(self, X, y):
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        score = self.model.score(X_test, y_test)
        print(f"Model R^2 Score: {score:.4f}")
        return score

    def predict(self, X):
        return self.model.predict(X)