"""
producer.py — SmartManuTech IoT Data Producer
Simula 5 máquinas industriales enviando métricas a Kafka cada 2 segundos.
Inyecta anomalías controladas (~5%) para probar la detección.
"""
import json
import random
import time
import datetime
import os
import logging
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "sensores-iot"

# Perfiles de operación normal por máquina (cada una tiene sus propios parámetros base)
MACHINE_PROFILES = {
    "MAQ-001": {"temp": 72, "vib": 2.3, "rpm": 1480, "kw": 42, "bar": 6.1},
    "MAQ-002": {"temp": 78, "vib": 2.7, "rpm": 1520, "kw": 48, "bar": 5.9},
    "MAQ-003": {"temp": 75, "vib": 2.5, "rpm": 1500, "kw": 45, "bar": 6.0},
    "MAQ-004": {"temp": 70, "vib": 2.1, "rpm": 1460, "kw": 40, "bar": 6.2},
    "MAQ-005": {"temp": 80, "vib": 3.0, "rpm": 1550, "kw": 52, "bar": 5.8},
}

TIPOS_ANOMALIA = [
    "sobrecalentamiento",
    "vibracion_excesiva",
    "sobrecarga_electrica",
    "caida_presion",
    "rpm_inestable",
]


def generar_dato(maquina_id: str) -> dict:
    p = MACHINE_PROFILES[maquina_id]
    dato = {
        "maquina_id": maquina_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "temperatura": round(random.gauss(p["temp"], 3), 2),
        "vibracion": round(random.gauss(p["vib"], 0.25), 3),
        "velocidad_rpm": round(random.gauss(p["rpm"], 70), 1),
        "consumo_kw": round(random.gauss(p["kw"], 4), 2),
        "presion_bar": round(random.gauss(p["bar"], 0.4), 2),
        "error_code": 0,
        "es_anomalia": False,
    }

    # Inyectar anomalía con 5% de probabilidad
    if random.random() < 0.05:
        tipo = random.choice(TIPOS_ANOMALIA)
        if tipo == "sobrecalentamiento":
            dato["temperatura"] += random.uniform(22, 45)
            dato["error_code"] = 1
        elif tipo == "vibracion_excesiva":
            dato["vibracion"] += random.uniform(2.5, 5.0)
            dato["error_code"] = 2
        elif tipo == "sobrecarga_electrica":
            dato["consumo_kw"] += random.uniform(28, 42)
            dato["error_code"] = 3
        elif tipo == "caida_presion":
            dato["presion_bar"] -= random.uniform(3.0, 4.5)
            dato["error_code"] = 4
        elif tipo == "rpm_inestable":
            dato["velocidad_rpm"] += random.choice([-1, 1]) * random.uniform(300, 600)
            dato["error_code"] = 5
        dato["es_anomalia"] = True

    return dato


def conectar_kafka(reintentos: int = 15, espera: int = 5) -> KafkaProducer:
    for intento in range(reintentos):
        try:
            prod = KafkaProducer(
                bootstrap_servers=KAFKA_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            logger.info(f"Kafka conectado en {KAFKA_SERVERS}")
            return prod
        except NoBrokersAvailable:
            logger.warning(f"Kafka no disponible, reintentando ({intento + 1}/{reintentos})...")
            time.sleep(espera)
    raise RuntimeError("No se pudo conectar a Kafka")


def main():
    producer = conectar_kafka()
    logger.info(f"Produciendo datos IoT en topic '{KAFKA_TOPIC}' — 5 máquinas cada 2 s")
    total = 0

    while True:
        for maquina_id in MACHINE_PROFILES:
            dato = generar_dato(maquina_id)
            producer.send(KAFKA_TOPIC, value=dato, key=maquina_id.encode())
            total += 1
            emoji = "⚠️ " if dato["es_anomalia"] else "✅"
            logger.info(
                f"{emoji} [{maquina_id}] "
                f"T={dato['temperatura']:.1f}°C | "
                f"V={dato['vibracion']:.2f}mm/s | "
                f"RPM={dato['velocidad_rpm']:.0f} | "
                f"kW={dato['consumo_kw']:.1f} | "
                f"bar={dato['presion_bar']:.2f}"
            )
        producer.flush()
        if total % 50 == 0:
            logger.info(f"--- Total mensajes enviados: {total} ---")
        time.sleep(2)


if __name__ == "__main__":
    main()