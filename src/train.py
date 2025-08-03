# src/train.py

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import argparse

def load_data(path):
    df = pd.read_csv(path)
    return train_test_split(df.drop("MedHouseVal", axis=1), df["MedHouseVal"], test_size=0.2, random_state=42)

def train_and_log(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # rmse = mean_squared_error(y_test, preds, squared=False)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, preds)

        # Log everything
        # mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(sk_model=model, name="model")

        print(f"{model_name} -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/housing.csv")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.input)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5)
    }

    for name, model in models.items():
        train_and_log(name, model, X_train, X_test, y_train, y_test)
