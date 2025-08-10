# api/main.py
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Header, Depends
from api.schema import BatchRequest
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from logging.handlers import RotatingFileHandler
from threading import Lock
from datetime import datetime
from time import perf_counter
import logging
import sqlite3
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import json
import os
import io

# =========================
# Config
# =========================
LOG_DIR = os.getenv("LOG_DIR", "logs")
DB_PATH = os.path.join(LOG_DIR, "predictions.db")
LOG_PATH = os.path.join(LOG_DIR, "predictions.log")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
API_KEY = os.getenv("API_KEY", "devkey")  # set a secret in env for prod

os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# Logging (independent of uvicorn)
# =========================
logger = logging.getLogger("housing_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =========================
# SQLite helpers
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                client_ip TEXT,
                payload TEXT NOT NULL,
                prediction TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms REAL NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def log_to_db(client_ip: str, payload: dict, prediction, status: str, latency_ms: float):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO logs (timestamp, client_ip, payload, prediction, status, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                client_ip,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(prediction, ensure_ascii=False),
                status,
                float(latency_ms),
            ),
        )
        conn.commit()
    finally:
        conn.close()

# =========================
# Load model (initial)
# =========================
model = None
current_model_version = 0
try:
    model = joblib.load(MODEL_PATH)
    current_model_version = 1
    logger.info(f"Loaded model from {MODEL_PATH} (version={current_model_version})")
except Exception as e:
    logger.exception(f"Failed to load model from {MODEL_PATH}: {e}")

# =========================
# FastAPI app
# =========================
app = FastAPI(title="California Housing API", version="1.0.0")

# =========================
# Prometheus metrics
# =========================
# Default HTTP metrics via Instrumentator (exposes /metrics)
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)

PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total number of prediction calls", ["status"]
)
PREDICTION_LATENCY_MS = Histogram(
    "prediction_latency_ms", "Prediction latency in milliseconds"
)
MODEL_VERSION = Gauge("model_version", "Current loaded model version")
RETRAINS_TOTAL = Counter("retrain_jobs_total", "Total number of successful model retrains", ["model_name"])
RETRAINS_TOTAL.labels(model_name="model.pkl").inc()


if current_model_version:
    MODEL_VERSION.set(current_model_version)

# =========================
# Dependencies
# =========================
def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

model_lock = Lock()

# =========================
# Startup
# =========================
@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("SQLite initialized at %s", DB_PATH)

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"message": "California Housing Model is Live 🚀", "version": current_model_version}

@app.get("/healthz")
def healthz():
    # model ok?
    model_ok = model is not None
    # db ok?
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False
    status = "ok" if (model_ok and db_ok) else "degraded"
    return {"status": status, "model_loaded": model_ok, "db_ok": db_ok, "model_version": current_model_version}

@app.post("/predict")
def predict(data: BatchRequest, request: Request):
    if model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    start = perf_counter()
    client_ip = (request.client.host if request and request.client else "unknown")

    # Build feature matrix
    X = [[
        d.MedInc, d.HouseAge, d.AveRooms, d.AveBedrms,
        d.Population, d.AveOccup, d.Latitude, d.Longitude
    ] for d in data.instances]

    payload_for_log = {"instances": [inst.dict() for inst in data.instances]}

    try:
        preds = model.predict(X).tolist()
        status = "success"
        return_payload = {"predictions": preds, "version": current_model_version}
        return return_payload
    except Exception as e:
        status = "error"
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    finally:
        latency_ms = (perf_counter() - start) * 1000.0

        # compact payload for file log
        short_payload = payload_for_log.copy()
        if len(short_payload["instances"]) > 3:
            short_payload["instances"] = short_payload["instances"][:3] + ["..."]

        logger.info(
            "client_ip=%s status=%s latency_ms=%.2f payload=%s",
            client_ip, status, latency_ms, json.dumps(short_payload, ensure_ascii=False),
        )

        # DB log
        try:
            pred_for_db = preds if status == "success" else {"error": True}
        except NameError:
            pred_for_db = {"error": True}
        log_to_db(client_ip, payload_for_log, pred_for_db, status, latency_ms)

        # Prometheus
        PREDICTIONS_TOTAL.labels(status=status).inc()
        PREDICTION_LATENCY_MS.observe(latency_ms)

# =========================
# Retrain utilities
# =========================
def retrain_from_dataframe(df: pd.DataFrame):
    if "MedHouseVal" not in df.columns:
        raise ValueError("Training CSV must include 'MedHouseVal' as target.")

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="DecisionTree_retrain"):
        model_new = DecisionTreeRegressor(max_depth=6, random_state=42)
        model_new.fit(X_train, y_train)
        preds = model_new.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", "DecisionTreeRegressor")
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model_new, "model")

    # persist atomically
    tmp_path = MODEL_PATH + ".tmp"
    joblib.dump(model_new, tmp_path)
    os.replace(tmp_path, MODEL_PATH)

    return {"rmse": float(rmse), "r2": float(r2), "model_path": MODEL_PATH}

RETRAIN_REQUESTS = Counter(
    "retrain_requests_total", "Retrain API requests", ["status"]
)
RETRAIN_LATENCY_MS = Histogram(
    "retrain_latency_ms", "Retrain latency in milliseconds"
)

@app.post("/retrain")
def retrain(file: UploadFile = File(...), _: bool = Depends(require_api_key)):
    t0 = perf_counter()
    status = "error"
    # read CSV into DataFrame
    try:
        content_bytes = file.file.read()
        df = pd.read_csv(io.StringIO(content_bytes.decode("utf-8")))
    except Exception as e:
        logger.exception(f"Failed to read uploaded CSV: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSV")

    # train and hot-reload
    global model, current_model_version
    with model_lock:
        try:
            result = retrain_from_dataframe(df)
            model = joblib.load(MODEL_PATH)
            current_model_version += 1
            MODEL_VERSION.set(current_model_version)
            RETRAINS_TOTAL.inc()
            status = "success"
            logger.info(
                "Model reloaded from %s, version=%d, metrics=%s",
                MODEL_PATH, current_model_version, result
            )
            return {"status": "ok", "version": current_model_version, "metrics": result}
        except Exception as e:
            logger.exception(f"Retrain failed: {e}")
            raise HTTPException(status_code=500, detail="Retrain failed")
        finally:
            RETRAIN_REQUESTS.labels(status=status).inc()
            RETRAIN_LATENCY_MS.observe((perf_counter() - t0) * 1000.0)
