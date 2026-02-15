FROM python:3.11-slim

WORKDIR /app

COPY inference/xgboost_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY training/feature_engineering.py /app/training/feature_engineering.py
COPY inference/xgboost_service/app.py /app/app.py
COPY models/xgboost_v1.json /app/models/xgboost_v1.json

ENV MODEL_PATH=/app/models/xgboost_v1.json
ENV MODEL_VERSION=xgboost_v1

EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --retries=3 CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
