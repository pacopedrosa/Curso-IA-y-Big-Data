"""
main.py — SmartManuTech REST API (FastAPI)
Expone los datos procesados de sensores IoT y anomalías detectadas.
Puerto: 8000
"""
import os
import time
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
from influxdb_client import InfluxDBClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuración ────────────────────────────────────────────────────────────
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "localhost")
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "smartmanutech-influx-token-2024")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "SmartManuTech")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "sensores_iot")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartManuTech IoT API",
    description="Monitorización de sensores industriales — BigData Aplicado",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Conexiones lazily inicializadas ──────────────────────────────────────────
_cass_session = None
_influx_client = None


def get_cassandra():
    global _cass_session
    if _cass_session is None:
        for _ in range(10):
            try:
                cluster = Cluster(
                    [CASSANDRA_HOST],
                    load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
                    protocol_version=4,
                    connect_timeout=20,
                )
                _cass_session = cluster.connect("smartmanutech")
                logger.info("Cassandra conectada")
                return _cass_session
            except Exception as e:
                logger.warning(f"Cassandra no disponible: {e}")
                time.sleep(4)
    return _cass_session


def get_influx():
    global _influx_client
    if _influx_client is None:
        _influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    return _influx_client


@app.on_event("startup")
async def startup():
    get_cassandra()
    get_influx()


# ── Modelos Pydantic ─────────────────────────────────────────────────────────
class Anomalia(BaseModel):
    id: str
    maquina_id: str
    timestamp: str
    tipo_anomalia: str
    campo: str
    valor: float
    umbral: float
    origen: str


class EstadoMaquina(BaseModel):
    maquina_id: str
    ultima_lectura: Optional[str] = None
    temperatura: Optional[float] = None
    vibracion: Optional[float] = None
    velocidad_rpm: Optional[float] = None
    consumo_kw: Optional[float] = None
    presion_bar: Optional[float] = None
    estado: str


class ResumenSistema(BaseModel):
    total_lecturas: int
    total_anomalias: int
    maquinas_monitorizadas: int


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "servicio": "SmartManuTech IoT API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    estado = {"api": "ok", "cassandra": "unknown", "influxdb": "unknown"}
    try:
        s = get_cassandra()
        s.execute("SELECT now() FROM system.local")
        estado["cassandra"] = "ok"
    except Exception:
        estado["cassandra"] = "error"
    try:
        get_influx().ping()
        estado["influxdb"] = "ok"
    except Exception:
        estado["influxdb"] = "error"
    return estado


@app.get("/anomalias/recientes", response_model=List[Anomalia], tags=["Anomalías"])
def get_anomalias_recientes(limite: int = Query(default=20, le=100, ge=1)):
    """Últimas anomalías detectadas en todas las máquinas."""
    try:
        rows = get_cassandra().execute(
            "SELECT id, maquina_id, timestamp, tipo_anomalia, campo, valor, umbral, origen "
            "FROM smartmanutech.anomalias LIMIT %s",
            (limite,),
        )
        return [
            Anomalia(
                id=str(r.id),
                maquina_id=r.maquina_id,
                timestamp=r.timestamp.isoformat() if r.timestamp else "",
                tipo_anomalia=r.tipo_anomalia or "",
                campo=r.campo or "",
                valor=r.valor or 0.0,
                umbral=r.umbral or 0.0,
                origen=r.origen or "",
            )
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=503, detail="Base de datos no disponible")


@app.get("/anomalias/maquina/{maquina_id}", response_model=List[Anomalia], tags=["Anomalías"])
def get_anomalias_maquina(maquina_id: str, limite: int = Query(default=10, le=50)):
    """Anomalías de una máquina específica."""
    try:
        rows = get_cassandra().execute(
            "SELECT id, maquina_id, timestamp, tipo_anomalia, campo, valor, umbral, origen "
            "FROM smartmanutech.anomalias WHERE maquina_id = %s ALLOW FILTERING LIMIT %s",
            (maquina_id, limite),
        )
        resultado = [
            Anomalia(
                id=str(r.id),
                maquina_id=r.maquina_id,
                timestamp=r.timestamp.isoformat() if r.timestamp else "",
                tipo_anomalia=r.tipo_anomalia or "",
                campo=r.campo or "",
                valor=r.valor or 0.0,
                umbral=r.umbral or 0.0,
                origen=r.origen or "",
            )
            for r in rows
        ]
        if not resultado:
            raise HTTPException(404, detail=f"Sin anomalías registradas para {maquina_id}")
        return resultado
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(503, detail="Error en base de datos")


@app.get("/maquina/{maquina_id}/estado", response_model=EstadoMaquina, tags=["Máquinas"])
def get_estado_maquina(maquina_id: str):
    """Estado actual de una máquina (última lectura)."""
    try:
        row = get_cassandra().execute(
            "SELECT * FROM smartmanutech.lecturas WHERE maquina_id = %s LIMIT 1",
            (maquina_id,),
        ).one()
        if not row:
            raise HTTPException(404, detail=f"Máquina {maquina_id} no encontrada")
        estado = "normal"
        if row.temperatura and row.temperatura > 95:
            estado = "critico"
        elif row.es_anomalia:
            estado = "advertencia"
        return EstadoMaquina(
            maquina_id=row.maquina_id,
            ultima_lectura=row.timestamp.isoformat() if row.timestamp else None,
            temperatura=row.temperatura,
            vibracion=row.vibracion,
            velocidad_rpm=row.velocidad_rpm,
            consumo_kw=row.consumo_kw,
            presion_bar=row.presion_bar,
            estado=estado,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(503, detail="Error en base de datos")


@app.get("/maquinas", tags=["Máquinas"])
def listar_maquinas():
    """Lista de máquinas monitorizadas."""
    return {
        "maquinas": [
            {"id": f"MAQ-{i:03d}", "nombre": f"Línea de producción {i}", "sector": f"Planta A — Sector {i}"}
            for i in range(1, 6)
        ]
    }


@app.get("/metricas/resumen", response_model=ResumenSistema, tags=["Métricas"])
def get_resumen():
    """Resumen global del sistema."""
    try:
        s = get_cassandra()
        total_anomalias = s.execute("SELECT COUNT(*) FROM smartmanutech.anomalias").one()[0]
        total_lecturas = s.execute("SELECT COUNT(*) FROM smartmanutech.lecturas").one()[0]
        return ResumenSistema(
            total_lecturas=int(total_lecturas),
            total_anomalias=int(total_anomalias),
            maquinas_monitorizadas=5,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(503, detail="Error en base de datos")


@app.get("/metricas/serie-temporal", tags=["Métricas"])
def get_serie_temporal(
    campo: str = Query(default="temperatura", description="temperatura|vibracion|velocidad_rpm|consumo_kw|presion_bar"),
    maquina_id: Optional[str] = Query(default=None, description="Ej: MAQ-001"),
    rango: str = Query(default="-10m", description="-5m | -1h | -24h"),
):
    """Serie temporal desde InfluxDB para un campo y máquina."""
    campos_validos = {"temperatura", "vibracion", "velocidad_rpm", "consumo_kw", "presion_bar"}
    if campo not in campos_validos:
        raise HTTPException(400, detail=f"Campo inválido. Opciones: {sorted(campos_validos)}")
    try:
        filtro = f'|> filter(fn: (r) => r["maquina_id"] == "{maquina_id}")' if maquina_id else ""
        flux = f"""
            from(bucket: "{INFLUXDB_BUCKET}")
              |> range(start: {rango})
              |> filter(fn: (r) => r["_measurement"] == "sensor_data")
              |> filter(fn: (r) => r["_field"] == "{campo}")
              {filtro}
              |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
        """
        tables = get_influx().query_api().query(flux)
        datos = [
            {
                "timestamp": rec.get_time().isoformat(),
                "maquina_id": rec.values.get("maquina_id", "?"),
                "valor": rec.get_value(),
            }
            for table in tables
            for rec in table.records
        ]
        return {"campo": campo, "total": len(datos), "datos": datos}
    except Exception as e:
        logger.error(f"Error InfluxDB: {e}")
        raise HTTPException(503, detail="Error en InfluxDB")