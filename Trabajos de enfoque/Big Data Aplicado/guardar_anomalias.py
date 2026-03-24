"""
guardar_anomalias.py — Script de prueba y utilidades
Permite verificar anomalías almacenadas en Cassandra desde línea de comandos.
Uso: python guardar_anomalias.py [limit]
"""
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CASS_HOST = "localhost"  # usa 'cassandra' dentro de Docker


def listar_anomalias(host: str = CASS_HOST, limit: int = 20):
    """Muestra las últimas anomalías almacenadas en Cassandra."""
    cluster = Cluster(
        [host],
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
        protocol_version=4,
    )
    try:
        session = cluster.connect("smartmanutech")
        rows = session.execute(
            f"SELECT maquina_id, timestamp, tipo_anomalia, campo, valor, umbral, origen "
            f"FROM smartmanutech.anomalias LIMIT {limit}"
        )
        print(f"\n{'='*70}")
        print(f" Últimas {limit} anomalías — SmartManuTech")
        print(f"{'='*70}")
        total = 0
        for r in rows:
            print(
                f"[{r.timestamp}] "
                f"{r.maquina_id:8s} | "
                f"{r.tipo_anomalia:30s} | "
                f"{r.campo}={r.valor:.2f} (umbral={r.umbral}) "
                f"[{r.origen}]"
            )
            total += 1
        print(f"{'='*70}")
        print(f"Total mostradas: {total}")
    except Exception as e:
        logger.error(f"Error al consultar Cassandra: {e}")
        logger.info("Asegúrate de que Cassandra está corriendo y el keyspace existe.")
        logger.info("Ejecuta primero: python guardar_cassandra.py")
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    listar_anomalias(limit=limit)