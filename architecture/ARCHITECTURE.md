# Self-Healing ML Microservice Mesh — Architecture

## System Overview

```mermaid
graph TB
    Client[Client / Load Test] -->|HTTP POST /api/v1/predict| GW[API Gateway<br/>FastAPI :8000]

    GW -->|Weighted Routing| XGB[XGBoost Service<br/>:8001]
    GW -->|Weighted Routing| PT[PyTorch Service<br/>:8002]
    GW -->|Fallback| FB[Fallback Service<br/>:8003]
    GW -.->|Shadow Mode| PT

    GW -->|Log Features| FS[Feature Store<br/>Redis :6379]
    GW -->|Expose /metrics| PROM[Prometheus<br/>:9090]

    PROM --> GRAF[Grafana<br/>:3000]

    FS --> DRIFT[Drift Detector]
    DRIFT -->|Trigger| RETRAIN[Prefect Retraining Flow]
    RETRAIN -->|Log & Register| MLF[MLflow<br/>:5000]
    RETRAIN -->|Deploy Canary| GW

    subgraph Inference Layer
        XGB
        PT
        FB
    end

    subgraph Observability
        PROM
        GRAF
    end

    subgraph ML Lifecycle
        MLF
        DRIFT
        RETRAIN
    end
```

## Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as API Gateway
    participant R as Router
    participant XGB as XGBoost Service
    participant PT as PyTorch Service
    participant FB as Fallback Service
    participant FS as Feature Store
    participant P as Prometheus

    C->>GW: POST /api/v1/predict
    GW->>FS: Log input features
    GW->>R: Route request (traffic split)

    alt Primary route (80%)
        R->>XGB: Forward request
        XGB-->>R: Prediction response
    else Canary route (20%)
        R->>PT: Forward request
        PT-->>R: Prediction response
    end

    alt Shadow mode enabled
        R-->>PT: Async shadow call (fire & forget)
        PT-->>FS: Log shadow prediction
    end

    alt Circuit breaker open
        R->>FB: Fallback to heuristic
        FB-->>R: Rule-based response
    end

    R-->>GW: Response
    GW->>P: Record latency, model version
    GW-->>C: JSON response
```

## Retraining & Rollback Flow

```mermaid
flowchart LR
    A[Scheduled / Drift Trigger] --> B[Check Drift Scores]
    B -->|Drift Detected| C[Generate Fresh Data]
    C --> D[Retrain XGBoost]
    D --> E[Log to MLflow]
    E --> F[Register New Model Version]
    F --> G[Deploy as Canary 10%]
    G --> H{Monitor Canary Metrics}
    H -->|Healthy| I[Promote to 100%]
    H -->|Degraded| J[Auto Rollback to Previous]
```

## Component Summary

| Component | Technology | Port | Purpose |
|-----------|-----------|------|---------|
| API Gateway | FastAPI | 8000 | Single entry point, routing, circuit breaker |
| XGBoost Service | FastAPI + XGBoost | 8001 | Primary ML model serving |
| PyTorch Service | FastAPI + PyTorch | 8002 | Secondary/canary model serving |
| Fallback Service | FastAPI | 8003 | Rule-based heuristic fallback |
| Feature Store | Redis | 6379 | Online feature storage & drift input |
| MLflow | MLflow Server | 5000 | Experiment tracking & model registry |
| Prometheus | Prometheus | 9090 | Metrics collection |
| Grafana | Grafana | 3000 | Dashboards & alerting |

## Failure Handling Strategy

1. **Service Failure**: Circuit breaker detects N consecutive failures → routes to fallback service
2. **Canary Degradation**: If canary model P95 latency or error rate exceeds threshold → auto-rollback to stable model
3. **Data Drift**: KS test / PSI exceeds threshold → triggers automated retraining pipeline
4. **Full Outage**: Fallback service provides rule-based predictions (no ML dependency)
5. **Recovery**: Health checks detect recovered services → gradually restore traffic routing
