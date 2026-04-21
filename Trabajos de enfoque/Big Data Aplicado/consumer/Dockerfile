FROM python:3.11-slim

WORKDIR /app

# Java (default-jdk-headless) necesario para PySpark MLlib
# gcc + libev para el driver de Cassandra
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libev4 libev-dev \
    default-jdk-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar consumer y el módulo ML
COPY consumer/consumer_anomalias.py .
COPY ml/ ml/

CMD ["python", "consumer_anomalias.py"]
