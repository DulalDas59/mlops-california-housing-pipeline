# src/train.py
import argparse
import json
import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    required_cols = {
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedHouseVal",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            "Expected 8 features + target 'MedHouseVal'."
        )
    return df


def split_xy(df: pd.DataFrame):
    y = df["MedHouseVal"]
    X = df.drop(columns=["MedHouseVal"])
    return X, y


def eval_regression(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)  # no 'squared' kw on old sklearn
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


def maybe_log_mlflow(
    enabled: bool,
    run_name: str,
    model_obj: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
):
    if not enabled:
        return
    try:
        import mlflow
        import mlflow.sklearn

        with mlflow.start_run(run_name=run_name):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(model_obj, "model")
    except Exception as e:
        # don't fail training if mlflow is misconfigured
        print(f"[WARN] MLflow logging skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train California Housing models and pick best.")
    parser.add_argument("--input", required=True, help="Path to input CSV (with MedHouseVal target).")
    parser.add_argument("--out", required=True, help="Path to save chosen model (joblib).")
    parser.add_argument("--metrics", required=True, help="Path to write metrics JSON.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size. Default 0.2")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-depth", type=int, default=6, help="DecisionTree max_depth.")
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="If set, log runs to MLflow (uses MLFLOW_TRACKING_URI if present).",
    )
    args = parser.parse_args()

    # 1) Load data
    df = load_dataset(args.input)
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 2) Train models
    results = {}

    # 2a) Linear Regression
    lr = LinearRegression(n_jobs=None)
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)
    lr_metrics = eval_regression(y_te, lr_pred)
    results["LinearRegression"] = {
        "params": {},
        "metrics": lr_metrics,
    }

    # 2b) Decision Tree
    dt = DecisionTreeRegressor(max_depth=args.max_depth, random_state=args.random_state)
    dt.fit(X_tr, y_tr)
    dt_pred = dt.predict(X_te)
    dt_metrics = eval_regression(y_te, dt_pred)
    results["DecisionTreeRegressor"] = {
        "params": {"max_depth": args.max_depth, "random_state": args.random_state},
        "metrics": dt_metrics,
    }

    # 3) Choose best by RMSE
    best_name = min(results.keys(), key=lambda k: results[k]["metrics"]["rmse"])
    best_model = {"LinearRegression": lr, "DecisionTreeRegressor": dt}[best_name]
    best_metrics = results[best_name]["metrics"]
    best_params = results[best_name]["params"]

    # 4) Persist model atomically
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp_out = args.out + ".tmp"
    joblib.dump(best_model, tmp_out)
    os.replace(tmp_out, args.out)

    # 5) Write metrics JSON (both models + chosen)
    payload = {
        "chosen_model": best_name,
        "chosen_metrics": best_metrics,
        "all_models": results,
    }
    with open(args.metrics, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"[INFO] Saved best model: {best_name} → {args.out}")
    print(f"[INFO] Metrics: {json.dumps(payload, indent=2)}")

    # 6) Optional MLflow logging
    maybe_log_mlflow(
        enabled=args.mlflow,
        run_name=f"{best_name}_california_housing",
        model_obj=best_model,
        params={"model": best_name, **best_params},
        metrics=best_metrics,
    )


if __name__ == "__main__":
    main()
