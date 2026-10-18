FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# dependencies first, so Docker caches this layer
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# only the files the API needs
COPY src/features.py src/predict.py src/app.py ./src/
COPY models/model.joblib ./models/model.joblib

# don't run as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8001"]