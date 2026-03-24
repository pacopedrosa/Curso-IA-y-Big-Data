"""
guardar_cassandra.py — Helper de Cassandra para SmartManuTech
Uso standalone: python guardar_cassandra.py
Inicializa el keyspace y las tablas en Cassandra.
"""
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CASS_HOST = "localhost"  # cambiar por 'cassandra' si se ejecuta dentro de Docker


def inicializar_schema(host: str = CASS_HOST):
    """Crea keyspace y tablas necesarias si no existen."""
    logger.info(f"Conectando a Cassandra en {host}...")
    cluster = Cluster(
        [host],
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
        protocol_version=4,
    )
    session = cluster.connect()

    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS smartmanutech
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    """)
    session.set_keyspace("smartmanutech")
    logger.info("Keyspace 'smartmanutech' creado/verificado")

    session.execute("""
        CREATE TABLE IF NOT EXISTS anomalias (
            id            UUID PRIMARY KEY,
            maquina_id    TEXT,
            timestamp     TIMESTAMP,
            tipo_anomalia TEXT,
            campo         TEXT,
            valor         DOUBLE,
            umbral        DOUBLE,
            origen        TEXT,
            temperatura   DOUBLE,
            vibracion     DOUBLE,
            velocidad_rpm DOUBLE,
            consumo_kw    DOUBLE,
            presion_bar   DOUBLE,
            error_code    INT
        )
    """)
    logger.info("Tabla 'anomalias' creada/verificada")

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
    logger.info("Tabla 'lecturas' creada/verificada")

    logger.info("Schema inicializado correctamente")
    cluster.shutdown()


# ── Notas sobre almacenamiento frío (AWS S3 / HDFS simulado) ─────────────────
# En un entorno de producción real, los datos históricos (>30 días) se moverían
# a almacenamiento frío. Opciones:
#
# AWS S3 (cloud):
#   import boto3
#   s3 = boto3.client('s3')
#   s3.put_object(Bucket='smartmanutech-historico', Key='2024/01/datos.json', Body=data)
#
# HDFS local (simulado con MinIO):
#   from minio import Minio
#   client = Minio('localhost:9000', 'minioadmin', 'minioadmin', secure=False)
#   client.put_object('sensores', 'lecturas/2024.json', data_stream, length)


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else CASS_HOST
    inicializar_schema(host)