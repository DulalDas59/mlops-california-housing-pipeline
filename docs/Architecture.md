# MLOps Pipeline – California Housing

## Architecture Overview

The diagram below shows the end-to-end architecture of our MLOps pipeline for the **California Housing Regression** project. It covers all stages — from data versioning and model training to deployment, monitoring, and retraining.

```mermaid
flowchart LR
    %% STYLE
    classDef store fill:#f0f8ff,stroke:#1976d2,stroke-width:1px
    classDef svc fill:#fff7e6,stroke:#ff9800,stroke-width:1px
    classDef ci fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    classDef obs fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px
    classDef user fill:#e3f2fd,stroke:#1565c0,stroke-width:1px

    %% ACTORS & SOURCES
    Dev["Developer<br/>Git + GitHub"]:::user
    Repo["GitHub Repo<br/><code>src/</code>, <code>api/</code>, <code>.github/</code>"]:::user

    %% DATA VERSIONING
    subgraph DVC[Data & Pipeline Versioning]
      DataRaw["Raw Data<br/>California Housing<br/><code>data/raw</code>"]:::store
      DataProc["Processed Features<br/><code>data/processed</code>"]:::store
      DVCRemote["DVC Remote Storage<br/>(e.g., S3/Drive)"]:::store
    end

    %% EXPERIMENTS & REGISTRY
    subgraph MLflow[MLflow Tracking & Registry]
      Tracking["Tracking Server<br/>params, metrics, artifacts"]:::store
      Registry["Model Registry<br/>Staging/Production"]:::store
    end

    %% CI/CD
    subgraph CI[GitHub Actions CI/CD]
      LintTest["Lint & Tests"]:::ci
      DvcPull["dvc pull & dvc repro"]:::ci
      BuildImg["Build Docker Image"]:::ci
      PushImg["Push to Docker Hub"]:::ci
      Deploy["Deploy (local/EC2)\n<code>docker run</code> / script"]:::ci
    end

    %% SERVING + OBSERVABILITY
    subgraph Serving[Model Serving & Monitoring]
      API["FastAPI Service<br/><code>/predict</code>, <code>/healthz</code>, <code>/retrain</code>, <code>/metrics</code>"]:::svc
      Logs["Structured Logs<br/>(files + SQLite)"]:::store
      Prom["Prometheus<br/>scrapes /metrics"]:::obs
      Graf["Grafana Dashboard"]:::obs
    end

    User["Client / App / Tester"]:::user

    %% FLOWS
    Dev -->|Push code & params| Repo
    Repo -->|Triggers| CI

    CI --> LintTest
    LintTest --> DvcPull
    DvcPull -->|Reproduce pipeline| DVC

    DataRaw <--> DVCRemote
    DataProc <--> DVCRemote

    DvcPull --> BuildImg --> PushImg --> Deploy --> API

    DvcPull -->|Train & Log| Tracking
    Tracking -->|Register Best| Registry
    Registry -->|Load model| API

    User -->|JSON payload| API
    API -->|Prediction JSON| User
    API -->|Requests, outputs, errors| Logs

    API -->|/metrics| Prom --> Graf

    %% Optional retraining trigger
    API -. secure /retrain .-> DvcPull
