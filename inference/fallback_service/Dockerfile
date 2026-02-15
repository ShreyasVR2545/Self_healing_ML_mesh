FROM python:3.11-slim

WORKDIR /app

COPY inference/fallback_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/fallback_service/app.py /app/app.py

EXPOSE 8003
HEALTHCHECK --interval=15s --timeout=5s --retries=3 CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
