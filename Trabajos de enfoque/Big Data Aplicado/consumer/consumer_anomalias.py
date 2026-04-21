"""
consumer_anomalias.py — SmartManuTech IoT Consumer
Pipeline de procesamiento en tiempo real:
  Kafka → Detección anomalías (umbrales + PySpark MLlib KMeans)
       → InfluxDB (series temporales para Grafana)
       → Cassandra (histórico de anomalías)
       → MinIO/S3 (almacenamiento distribuido histórico en Parquet)
"""
import json
import os
import sys
import logging
import time
import uuid
import io
from datetime import datetime, timezone

import boto3
import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy

sys.path.insert(0, "/app")
from ml.predictive_model import DetectorAnomalias

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuración desde variables de entorno ─────────────────────────────────
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "localhost")
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "smartmanutech-influx-token-2024")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "SmartManuTech")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "sensores_iot")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "smartmanutech")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "smartmanutech123")
MINIO_BUCKET_HISTORICAL = "smartmanutech-historical"
MINIO_BUCKET_ANOMALIAS = "smartmanutech-anomalias"
KAFKA_TOPIC = "sensores-iot"
MINIO_FLUSH_INTERVAL = 100  # filas antes de escribir parquet a MinIO

# ── Umbrales de detección por reglas de negocio ──────────────────────────────
UMBRALES = {
    "temperatura":  {"max": 95.0,  "unidad": "°C"},
    "vibracion":    {"max": 4.5,   "unidad": "mm/s"},
    "consumo_kw":   {"max": 70.0,  "unidad": "kW"},
    "presion_bar":  {"min": 2.5,   "unidad": "bar"},
}


# ── Cassandra ────────────────────────────────────────────────────────────────
def iniciar_cassandra(host: str, intentos: int = 12) -> object:
    for i in range(intentos):
        try:
            cluster = Cluster(
                [host],
                load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
                protocol_version=4,
                connect_timeout=20,
            )
            session = cluster.connect()
            _crear_esquema(session)
            logger.info(f"Cassandra conectada en {host}")
            return session
        except Exception as e:
            logger.warning(f"Cassandra no disponible ({i + 1}/{intentos}): {e}")
            time.sleep(8)
    raise RuntimeError("No se pudo conectar a Cassandra")


def _crear_esquema(session):
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS smartmanutech
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    """)
    session.set_keyspace("smartmanutech")
    session.execute("""
        CREATE TABLE IF NOT EXISTS anomalias (
            id          UUID PRIMARY KEY,
            maquina_id  TEXT,
            timestamp   TIMESTAMP,
            tipo_anomalia TEXT,
            campo       TEXT,
            valor       DOUBLE,
            umbral      DOUBLE,
            origen      TEXT,
            temperatura DOUBLE,
            vibracion   DOUBLE,
            velocidad_rpm DOUBLE,
            consumo_kw  DOUBLE,
            presion_bar DOUBLE,
            error_code  INT
        )
    """)
    session.execute("""
        CREATE TABLE IF NOT EXISTS lecturas (
            maquina_id    TEXT,
            timestamp     TIMESTAMP,
            temperatura   DOUBLE,
            vibracion     DOUBLE,
            velocidad_rpm DOUBLE,
            consumo_kw    DOUBLE,
            presion_bar   DOUBLE,
            error_code    INT,
            es_anomalia   BOOLEAN,
            PRIMARY KEY (maquina_id, timestamp)
        ) WITH CLUSTERING ORDER BY (timestamp DESC)
    """)
    logger.info("Esquema Cassandra creado/verificado")


def guardar_anomalia(session, dato: dict, tipo: str, campo: str, valor: float, umbral: float, origen: str):
    session.execute("""
        INSERT INTO smartmanutech.anomalias
        (id, maquina_id, timestamp, tipo_anomalia, campo, valor, umbral, origen,
         temperatura, vibracion, velocidad_rpm, consumo_kw, presion_bar, error_code)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        uuid.uuid4(),
        dato["maquina_id"],
        datetime.fromisoformat(dato["timestamp"]),
        tipo, campo,
        float(valor), float(umbral), origen,
        float(dato.get("temperatura", 0)),
        float(dato.get("vibracion", 0)),
        float(dato.get("velocidad_rpm", 0)),
        float(dato.get("consumo_kw", 0)),
        float(dato.get("presion_bar", 6.0)),
        int(dato.get("error_code", 0)),
    ))


def guardar_lectura(session, dato: dict, es_anomalia: bool):
    session.execute("""
        INSERT INTO smartmanutech.lecturas
        (maquina_id, timestamp, temperatura, vibracion, velocidad_rpm,
         consumo_kw, presion_bar, error_code, es_anomalia)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        dato["maquina_id"],
        datetime.fromisoformat(dato["timestamp"]),
        float(dato.get("temperatura", 0)),
        float(dato.get("vibracion", 0)),
        float(dato.get("velocidad_rpm", 0)),
        float(dato.get("consumo_kw", 0)),
        float(dato.get("presion_bar", 6.0)),
        int(dato.get("error_code", 0)),
        es_anomalia,
    ))


# ── InfluxDB ─────────────────────────────────────────────────────────────────
def iniciar_influxdb(url, token, org):
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    logger.info(f"InfluxDB conectada en {url}")
    return write_api


def escribir_influxdb(write_api, dato: dict, es_anomalia: bool):
    point = (
        Point("sensor_data")
        .tag("maquina_id", dato["maquina_id"])
        .tag("es_anomalia", str(es_anomalia).lower())
        .field("temperatura",    float(dato.get("temperatura", 0)))
        .field("vibracion",      float(dato.get("vibracion", 0)))
        .field("velocidad_rpm",  float(dato.get("velocidad_rpm", 0)))
        .field("consumo_kw",     float(dato.get("consumo_kw", 0)))
        .field("presion_bar",    float(dato.get("presion_bar", 6.0)))
        .field("error_code",     int(dato.get("error_code", 0)))
        .field("anomalia_flag",  1 if es_anomalia else 0)
    )
    write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)


# ── MinIO (almacenamiento distribuido S3-compatible) ──────────────────────────
def iniciar_minio() -> boto3.client:
    """Inicializa el cliente boto3 apuntando a MinIO y crea los buckets."""
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )
    for bucket in [MINIO_BUCKET_HISTORICAL, MINIO_BUCKET_ANOMALIAS]:
        try:
            s3.create_bucket(Bucket=bucket)
            logger.info(f"MinIO: bucket '{bucket}' creado")
        except Exception:
            pass  # bucket ya existe
    logger.info(f"MinIO conectado en {MINIO_ENDPOINT}")
    return s3


def flush_minio(s3_client, filas: list, bucket: str, prefijo: str):
    """Escribe una lista de registros como archivo Parquet en MinIO."""
    if not filas or s3_client is None:
        return
    try:
        df = pd.DataFrame(filas)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        key = f"{prefijo}/{ts}.parquet"
        s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"MinIO: {len(filas)} registros → s3://{bucket}/{key}")
    except Exception as e:
        logger.warning(f"Error escribiendo en MinIO: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 55)
    logger.info(" SmartManuTech Consumer — Iniciando pipeline IoT ")
    logger.info("=" * 55)

    cass = iniciar_cassandra(CASSANDRA_HOST)
    write_api = iniciar_influxdb(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG)
    s3_client = iniciar_minio()
    detector = DetectorAnomalias()
    logger.info("Modelo PySpark MLlib KMeans listo (entrenado con histórico simulado)")

    # Conectar a Kafka con reintentos
    consumer = None
    for intento in range(15):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="smartmanutech-group-v1",
            )
            logger.info(f"Kafka conectado — escuchando topic '{KAFKA_TOPIC}'")
            break
        except NoBrokersAvailable:
            logger.warning(f"Kafka no disponible ({intento + 1}/15), esperando...")
            time.sleep(5)
    if consumer is None:
        raise RuntimeError("No se pudo conectar a Kafka")

    stats = {"total": 0, "anomalias_umbral": 0, "anomalias_ml": 0}
    batch_lecturas: list = []
    batch_anomalias: list = []

    for mensaje in consumer:
        dato = mensaje.value
        stats["total"] += 1
        es_anomalia = False

        # 1) Detección por umbrales (reglas de negocio)
        for campo, cfg in UMBRALES.items():
            valor = float(dato.get(campo, 0))
            tipo = None
            umbral_val = None
            if "max" in cfg and valor > cfg["max"]:
                tipo = f"UMBRAL_ALTO_{campo.upper()}"
                umbral_val = cfg["max"]
            elif "min" in cfg and valor < cfg["min"]:
                tipo = f"UMBRAL_BAJO_{campo.upper()}"
                umbral_val = cfg["min"]
            if tipo:
                guardar_anomalia(cass, dato, tipo, campo, valor, umbral_val, "umbral")
                stats["anomalias_umbral"] += 1
                es_anomalia = True
                logger.warning(
                    f"⚠️  UMBRAL [{dato['maquina_id']}] {campo}={valor:.2f}{cfg['unidad']} "
                    f"(límite {umbral_val})"
                )

        # 2) Detección ML — PySpark MLlib KMeans (distancia al centroide)
        if detector.predecir(dato):
            if not es_anomalia:  # evitar duplicar si ya captado por umbral
                guardar_anomalia(
                    cass, dato, "ANOMALIA_ML", "multivariante",
                    detector.ultimo_score, detector.threshold or 0.0, "kmeans_spark"
                )
                stats["anomalias_ml"] += 1
                logger.warning(
                    f"🤖 ML [{dato['maquina_id']}] dist={detector.ultimo_score:.3f} — anomalía multivariante"
                )
            es_anomalia = True

        # 3) Persistir en InfluxDB (TODOS los datos para Grafana)
        escribir_influxdb(write_api, dato, es_anomalia)

        # 4) Guardar lectura histórica en Cassandra
        guardar_lectura(cass, dato, es_anomalia)

        # 5) Acumular en batch para MinIO (Parquet, almacenamiento distribuido)
        fila = {**dato, "es_anomalia": es_anomalia}
        batch_lecturas.append(fila)
        if es_anomalia:
            batch_anomalias.append(fila)

        # Flush a MinIO cada MINIO_FLUSH_INTERVAL lecturas
        if len(batch_lecturas) >= MINIO_FLUSH_INTERVAL:
            flush_minio(s3_client, batch_lecturas, MINIO_BUCKET_HISTORICAL, "lecturas")
            batch_lecturas = []
        if len(batch_anomalias) >= 10:
            flush_minio(s3_client, batch_anomalias, MINIO_BUCKET_ANOMALIAS, "anomalias")
            batch_anomalias = []

        if stats["total"] % 25 == 0:
            logger.info(
                f"📊 Procesados: {stats['total']} | "
                f"Umbral: {stats['anomalias_umbral']} | "
                f"ML: {stats['anomalias_ml']}"
            )


if __name__ == "__main__":
    main()