FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema para Cassandra driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libev4 libev-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# La API no necesita PySpark ni pyarrow completos — instalar solo lo necesario
RUN pip install --no-cache-dir \
    kafka-python==2.0.2 \
    cassandra-driver==3.28.0 \
    influxdb-client==1.36.1 \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    numpy==1.26.2 \
    pandas==2.1.3 \
    boto3==1.34.0

COPY api/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
