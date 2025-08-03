# src/data_loader.py

from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def save_data(path="data/raw/housing.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    df.to_csv(path, index=False)
    print(f"Dataset saved to: {path}")

if __name__ == "__main__":
    save_data()
