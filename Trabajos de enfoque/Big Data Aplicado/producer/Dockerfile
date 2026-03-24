FROM python:3.11-slim

WORKDIR /app

# Instalar sólo lo necesario para el productor
RUN pip install --no-cache-dir kafka-python==2.0.2

COPY producer/producer.py .

CMD ["python", "producer.py"]
