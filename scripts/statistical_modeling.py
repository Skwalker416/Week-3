import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    X = data.drop(['TotalClaims'], axis=1)
    y = data['TotalClaims']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'XGBoost': xgb.XGBRegressor()
    }
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = model.fit(X_train, y_train)
    return trained_models

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"{name}: MSE = {mse:.2f}, R^2 = {r2:.2f}")

# Example Usage
if __name__ == "__main__":
    data = load_data('../data/preprocessed_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    trained_models = train_models(X_train, y_train)
    evaluate_models(trained_models, X_test, y_test)
