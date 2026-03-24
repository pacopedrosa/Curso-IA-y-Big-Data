# SmartManuTech — Plataforma Big Data IoT Industrial

Sistema de procesamiento en tiempo real de sensores industriales con detección de anomalías,
almacenamiento escalable y visualización en dashboard Grafana.

## Arquitectura

```
[5 Máquinas IoT]
      │  (datos cada 2s)
      ▼
 Apache Kafka  ─────────────────────────────────────┐
 (topic: sensores-iot)                              │
      │                                             │
      ▼                                             ▼
 Consumer Python                            API FastAPI (:8000)
 ├── Umbrales (reglas)                       ├── /anomalias/recientes
 ├── Isolation Forest (ML)                   ├── /maquina/{id}/estado
 ├── → InfluxDB (series temporales)          ├── /metricas/resumen
 └── → Cassandra (histórico anomalías)       └── /metricas/serie-temporal
           │
           ▼
        Grafana (:3005)
        Dashboard en tiempo real
```

## Stack tecnológico

| Componente    | Tecnología               | Puerto |
|---------------|--------------------------|--------|
| Mensajería    | Apache Kafka + Zookeeper | 9092   |
| Base de datos | Apache Cassandra 4.1     | 9042   |
| Series temp.  | InfluxDB 2.7             | 8086   |
| Visualización | Grafana 10.2             | 3005   |
| API REST      | FastAPI + Uvicorn        | 8000   |
| ML            | Isolation Forest (sklearn)| —     |

## Inicio rápido

### 1. Levantar toda la infraestructura

```bash
cd "Big Data Aplicado"
docker compose up --build -d
```

El primer arranque puede tardar 3-5 minutos mientras Cassandra e InfluxDB se inicializan.

### 2. Verificar que los servicios están activos

```bash
docker compose ps
```

Todos deben aparecer como `healthy` o `running`.

### 3. Ver logs del pipeline

```bash
# Ver productor IoT enviando datos
docker logs -f bigdata-producer

# Ver consumer detectando anomalías
docker logs -f bigdata-consumer

# Ver la API
docker logs -f bigdata-api
```

### 4. Acceder a Grafana

URL: http://localhost:3005
Usuario: `admin`  |  Contraseña: `admin123`

El dashboard **SmartManuTech — IoT Dashboard** se carga automáticamente.

### 5. Probar la API

```bash
# Health check
curl http://localhost:8000/health

# Últimas anomalías
curl http://localhost:8000/anomalias/recientes

# Estado de una máquina
curl http://localhost:8000/maquina/MAQ-001/estado

# Resumen del sistema
curl http://localhost:8000/metricas/resumen

# Serie temporal de temperatura (últimos 10 min)
curl "http://localhost:8000/metricas/serie-temporal?campo=temperatura&rango=-10m"
```

Documentación interactiva de la API: http://localhost:8000/docs

### 6. Apagar el sistema

```bash
docker compose down
# Para borrar también los datos persistentes:
docker compose down -v
```

## Estructura de archivos

```
Big Data Aplicado/
├── docker-compose.yml        # 7 servicios orquestados
├── requirements.txt          # Dependencias Python
├── README.md
├── DOCUMENTO_FINAL.md        # Informe académico
│
├── producer/
│   ├── producer.py           # Simulador de 5 máquinas IoT
│   └── Dockerfile
│
├── consumer/
│   ├── consumer_anomalias.py # Pipeline: Kafka → ML → Cassandra + InfluxDB
│   └── Dockerfile
│
├── ml/
│   ├── __init__.py
│   └── predictive_model.py   # Isolation Forest (sklearn)
│
├── api/
│   ├── main.py               # FastAPI REST endpoints
│   └── Dockerfile
│
├── dashboard/
│   ├── grafana-datasource.yml    # Datasource InfluxDB auto-provisionado
│   ├── grafana-dashboard.yml     # Config provisioning Grafana
│   └── smartmanutech-dashboard.json  # Dashboard JSON
│
├── guardar_cassandra.py      # Helper: inicializar schema Cassandra
└── guardar_anomalias.py      # Utilidad: listar anomalías guardadas
```

## Variables de entorno (referencia)

| Variable                   | Valor por defecto               |
|----------------------------|---------------------------------|
| KAFKA_BOOTSTRAP_SERVERS    | kafka:29092                     |
| CASSANDRA_HOST             | cassandra                       |
| INFLUXDB_URL               | http://influxdb:8086            |
| INFLUXDB_TOKEN             | smartmanutech-influx-token-2024 |
| INFLUXDB_ORG               | SmartManuTech                   |
| INFLUXDB_BUCKET            | sensores_iot                    |

## Detección de anomalías

El sistema usa **dos capas** de detección:

1. **Umbrales de negocio** (reglas deterministas):
   - Temperatura > 95 °C
   - Vibración > 4.5 mm/s
   - Consumo > 70 kW
   - Presión < 2.5 bar

2. **Isolation Forest** (ML no supervisado):
   - Entrenado con 2.000 muestras históricas simuladas de operación normal
   - Detecta combinaciones anómalas que los umbrales individuales no capturan
   - Reentrenamiento online cada 500 nuevas muestras (adaptación a concept drift)
