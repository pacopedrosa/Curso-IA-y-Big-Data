FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema para Cassandra driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libev4 libev-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar consumer y el módulo ML
COPY consumer/consumer_anomalias.py .
COPY ml/ ml/

CMD ["python", "consumer_anomalias.py"]
