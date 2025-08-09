# MLOps Pipeline — California Housing with FastAPI

## Part 1: Repository & Data Versioning

- Created GitHub repo **mlops-california-housing-pipeline**.
- Set up project structure:

```bash
/data      → Dataset storage  
/models    → Saved models  
/api       → FastAPI app  
/notebooks → EDA/training notebooks  
requirements.txt  
```
- Loaded & preprocessed **California Housing** dataset.
- Added `.idea` to repo as per request.
- Initialized **DVC** (optional for dataset tracking).
- Noted: `dvc.yaml` is generated when you run `dvc run` or `dvc stage add`.

---

## Part 2: Model Development & Experiment Tracking

- Trained two models:
  - **Linear Regression**
  - **Decision Tree Regressor**
- Used **MLflow** to:
  - Track parameters, metrics (MAE, RMSE, R²)
  - Log models & artifacts
  - Register best model in **MLflow Model Registry**.

---

## Part 3: API & Docker Packaging

- Built **FastAPI** prediction API (`main.py`):
  - `/predict` endpoint accepts JSON and returns prediction.
  - Integrated **Pydantic** for input validation.
- Containerized API with **Docker**:
  - Dockerfile created with Python 3.10-slim.
  - Installed dependencies via `requirements.txt`.
  - Used `uvicorn` for serving.
  - Exposed on port **8000**.

---

## Part 4: CI/CD with GitHub Actions

- Created GitHub Actions workflow:
  - Linting & testing on push.
  - Build Docker image & push to Docker Hub.
  - Deploy container locally or on EC2 via `docker run`.

---

## Part 5: Logging & Monitoring

- Added **logging**:
  - Logs incoming requests & predictions to `logs/predictions.log`.
  - Also logs to **SQLite** database (`predictions.db`).
- Integrated **Prometheus** metrics:
  - Tracked `http_requests_total`, `latency`, `model_version`, and `predictions_total`.
- Built **Grafana dashboard**:
  - Panels for Model Version, Requests/sec, Retrains, Latency, Predictions by Status.
  - Updated JSON to make **charts bigger** for better visibility.

---

## Bonus Features

- **Input validation** with Pydantic schema.
- **Prometheus + Grafana** integration with sample dashboard.
- Placeholder **model retraining trigger** endpoint to simulate retraining when new data arrives.

---

## Current Artifacts

- **GitHub Repo**: Contains source code, Dockerfile, requirements, CI/CD workflow.
- **Docker Image**: Containerized API with logging, monitoring.
- **Grafana Dashboard JSON**: Configured for bigger charts.
- **MLflow Tracking**: Runs & models logged.
- **SQLite Logs**: For predictions and monitoring.
