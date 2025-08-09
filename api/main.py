# api/main.py
from fastapi import FastAPI, Request, HTTPException
from api.schema import BatchRequest
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import joblib
import logging
from logging.handlers import RotatingFileHandler
import os
import sqlite3
import json
from datetime import datetime
from time import perf_counter

# ---------- Config ----------
LOG_DIR = os.getenv("LOG_DIR", "logs")
DB_PATH = os.path.join(LOG_DIR, "predictions.db")
LOG_PATH = os.path.join(LOG_DIR, "predictions.log")
MODEL_PATH = os.getenv("MODEL_PATH", "models/dt.pkl")

os.makedirs(LOG_DIR, exist_ok=True)

# ---------- Logger (file + console, independent of uvicorn) ----------
logger = logging.getLogger("housing_api")
logger.setLevel(logging.INFO)
# Avoid duplicated handlers if code reloads
logger.handlers.clear()

file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ---------- SQLite helpers ----------
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

# ---------- Load model ----------
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    logger.exception(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None  # healthz will report failure

# ---------- FastAPI app ----------
app = FastAPI(title="California Housing API", version="1.0.0")

# ---------- Prometheus metrics ----------
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of prediction calls",
    ["status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds",
)

# instrument default HTTP metrics + expose /metrics
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("SQLite initialized at %s", DB_PATH)

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "California Housing Model is Live 🚀"}

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
    return {"status": status, "model_loaded": model_ok, "db_ok": db_ok}

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
        return_payload = {"predictions": preds}
        return return_payload
    except Exception as e:
        status = "error"
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    finally:
        latency_ms = (perf_counter() - start) * 1000.0

        # file log (one line per request, not per row)
        short_payload = payload_for_log.copy()
        # keep logs compact
        if len(short_payload["instances"]) > 3:
            short_payload["instances"] = short_payload["instances"][:3] + ["..."]
        logger.info(
            "client_ip=%s status=%s latency_ms=%.2f payload=%s",
            client_ip, status, latency_ms, json.dumps(short_payload, ensure_ascii=False),
        )

        # DB log (full payload + predictions if success)
        try:
            pred_for_db = preds if status == "success" else {"error": True}
        except NameError:
            pred_for_db = {"error": True}
        log_to_db(client_ip, payload_for_log, pred_for_db, status, latency_ms)

        # Prometheus
        PREDICTIONS_TOTAL.labels(status=status).inc()
        PREDICTION_LATENCY.observe(latency_ms)
