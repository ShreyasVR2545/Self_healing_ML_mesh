FROM python:3.11-slim

WORKDIR /app

COPY inference/pytorch_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY training/feature_engineering.py /app/training/feature_engineering.py
COPY inference/pytorch_service/app.py /app/app.py
COPY models/pytorch_v1.pt /app/models/pytorch_v1.pt

ENV MODEL_PATH=/app/models/pytorch_v1.pt
ENV MODEL_VERSION=pytorch_v1

EXPOSE 8002
HEALTHCHECK --interval=15s --timeout=5s --retries=3 CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
