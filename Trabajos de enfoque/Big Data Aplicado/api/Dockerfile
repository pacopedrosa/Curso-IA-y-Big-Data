FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema para Cassandra driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libev4 libev-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
