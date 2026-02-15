FROM python:3.11-slim

WORKDIR /app

COPY gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/app.py /app/app.py
COPY gateway/router.py /app/router.py
COPY gateway/shadow.py /app/shadow.py
COPY gateway/rollback.py /app/rollback.py

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
