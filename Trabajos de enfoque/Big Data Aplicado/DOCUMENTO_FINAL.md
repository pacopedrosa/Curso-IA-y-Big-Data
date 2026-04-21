# SmartManuTech: Plataforma Big Data para Detección de Anomalías en Líneas de Producción Industrial

**Asignatura:** Big Data Aplicado  
**Grado:** Técnico Superior en Inteligencia Artificial y Big Data  
**Fecha:** Marzo 2026  

---

## 1. Introducción

La Industria 4.0 ha transformado radicalmente la forma en que las empresas manufactureras gestionan sus procesos productivos. La proliferación de sensores IoT (*Internet of Things*) en entornos industriales genera volúmenes de datos que, correctamente procesados, pueden convertirse en una ventaja competitiva real: la capacidad de predecir fallos antes de que ocurran.

SmartManuTech es una empresa ficticia de manufactura que utiliza sensores IoT en sus líneas de producción pero carece de la capacidad técnica para procesar esos datos en tiempo real. El resultado es que los fallos se detectan tarde, el mantenimiento es reactivo y los costes de parada no planificada son elevados.

Este proyecto desarrolla una plataforma Big Data completa que resuelve exactamente ese problema: un sistema de ingesta, procesamiento, almacenamiento distribuido, aprendizaje automático y visualización que opera en tiempo real sobre datos industriales simulados.

El sistema implementado es funcional y ejecutable mediante Docker, lo que permite reproducir el entorno completo en cualquier máquina con un solo comando.

---

## 2. Objetivos

### Objetivo principal

Diseñar e implementar una plataforma Big Data que permita a SmartManuTech procesar datos de sensores industriales en tiempo real, detectar anomalías de forma automática y visualizar el estado de la planta en un dashboard operativo.

### Objetivos específicos

1. **Procesamiento en tiempo real:** Ingesta y procesamiento de datos IoT con latencia inferior a 5 segundos.
2. **Detección de anomalías:**  Implementar un sistema de doble capa: umbrales de negocio (reglas deterministas) + modelo de Machine Learning no supervisado (Isolation Forest).
3. **Almacenamiento escalable:** Persistencia en dos bases de datos complementarias: Cassandra (histórico de anomalías) e InfluxDB (series temporales para visualización).
4. **Visualización operativa:** Dashboard en Grafana que permita a operarios de planta monitorizar el estado de las máquinas en tiempo real.
5. **API REST:** Exposición de los datos procesados a través de endpoints documentados para integración con otros sistemas.

---

## 3. Análisis del problema — SmartManuTech

### 3.1 Situación actual

SmartManuTech opera 5 líneas de producción (MAQ-001 a MAQ-005), cada una equipada con sensores que registran los siguientes parámetros cada dos segundos:

| Sensor              | Unidad | Rango normal              | Umbral de alerta |
|---------------------|--------|---------------------------|-----------------|
| Temperatura         | °C     | 70–82 °C (según máquina)  | > 95 °C         |
| Vibración mecánica  | mm/s   | 2,1–3,0 mm/s              | > 4,5 mm/s      |
| Velocidad de giro   | RPM    | 1460–1550 RPM             | Desviación >25% |
| Consumo eléctrico   | kW     | 40–52 kW                  | > 70 kW         |
| Presión de circuito | bar    | 5,8–6,2 bar               | < 2,5 bar       |
| Código de error     | enum   | 0 (sin error)             | > 0             |

El volumen generado es de aproximadamente 10 lecturas por segundo (5 máquinas × 1 lectura cada 2 s), lo que supone 864.000 registros diarios. Sin un sistema de procesamiento automatizado, un operario humano no puede analizar este flujo.

### 3.2 Problemas identificados

- **Detección tardía de fallos:** Los fallos se detectan al manifestarse físicamente, cuando ya hay daño.
- **Datos en silos:** Cada sensor guarda sus datos de forma independiente; no hay visión integrada.
- **Sin análisis predictivo:** El mantenimiento es 100% reactivo o se basa únicamente en calendario.
- **Escalabilidad limitada:** Si se amplía el número de máquinas, el sistema actual no escala.

### 3.3 Tipos de anomalías a detectar

1. **Sobrecalentamiento** — Temperatura supera 95 °C. Causa: fallo de refrigeración o rodamientos.
2. **Vibración excesiva** — Vibración > 4,5 mm/s. Causa: desequilibrio, desgaste de rodamientos.
3. **Sobrecarga eléctrica** — Consumo > 70 kW. Causa: atasco mecánico o sobrecarga.
4. **Caída de presión** — Presión < 2,5 bar. Causa: fuga en circuito hidráulico.
5. **RPM inestables** — Desviación > 25% del nominal. Causa: fallo del variador de frecuencia.
6. **Anomalías multivariantes** — Combinaciones inusuales que no disparan umbrales individuales pero son estadísticamente anómalas (detectadas por ML).

---

## 4. Desarrollo del sistema

### 4.1 Arquitectura de la solución

El sistema sigue el patrón Lambda simplificado, con una capa de velocidad (tiempo real) y una capa de almacenamiento (persistencia):

```
┌─────────────────────────────────────────────────────────────────┐
│                     CAPA DE INGESTA                             │
│                                                                 │
│   [MAQ-001] [MAQ-002] [MAQ-003] [MAQ-004] [MAQ-005]            │
│        │         │         │         │         │               │
│        └─────────┴────┬────┴─────────┘         │               │
│                       ▼                                        │
│              Producer Python                                    │
│              (producer.py)                                      │
│                       │                                        │
│                       ▼                                        │
│              Apache Kafka                                       │
│              topic: sensores-iot                                │
└───────────────────────┼─────────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────────┐
│                CAPA DE PROCESAMIENTO                            │
│                       ▼                                        │
│              Consumer Python                                    │
│              ┌────────┴────────┐                               │
│              │                 │                               │
│         Umbrales           PySpark MLlib                       │
│         de negocio         KMeans (k=2)                        │
│              │                 │                               │
│              └────────┬────────┘                               │
│                       │                                        │
│              ¿Es anomalía?                                      │
│             (umbral OR ML)                                      │
└───────────────────────┼─────────────────────────────────────────┘
                        │
        ┌───────────────┼──────────────────────┐
        ▼               ▼                      ▼
┌──────────────┐ ┌─────────────────┐  ┌──────────────────┐
│  InfluxDB    │ │   Cassandra      │  │  MinIO (S3)      │
│  (todos los  │ │   (solo          │  │  Parquet         │
│   datos,     │ │   anomalías +    │  │  histórico       │
│   series     │ │   histórico)     │  │  distribuido     │
│   temporales)│ │                  │  │  :9000/:9001     │
└──────┬───────┘ └────────┬─────────┘  └──────────────────┘
       │                  │
       ▼                  ▼
  ┌─────────┐       ┌──────────┐
  │ Grafana │       │  FastAPI │
  │ :3005   │       │  :8000   │
  └─────────┘       └──────────┘
```

### 4.2 Descripción de cada componente

#### Apache Kafka (mensajería)

Kafka actúa como *bus de eventos* central. El productor envía mensajes JSON al topic `sensores-iot`. Kafka garantiza que los mensajes se entreguen al consumidor aunque haya una desconexión temporal. Con `KAFKA_LOG_RETENTION_HOURS: 24`, los datos se conservan 24 horas y permiten reprocesar en caso de fallo.

#### Consumer Python (procesamiento en tiempo real)

El consumidor es el corazón del pipeline. Para cada mensaje recibido:

1. Lee el JSON del topic Kafka.
2. Aplica umbrales de negocio (5 condiciones).
3. Aplica Isolation Forest (predicción ML).
4. Si hay anomalía → guarda en Cassandra tabla `anomalias`.
5. Siempre → guarda en InfluxDB para visualización.
6. Siempre → guarda en Cassandra tabla `lecturas` (histórico completo).

#### Apache Cassandra (base de datos escalable)

Cassandra es una base de datos NoSQL distribuida y orientada a columnas, diseñada para escrituras masivas y alta disponibilidad. En este proyecto almacena:

- **Tabla `anomalias`**: ID, máquina, timestamp, tipo, campo, valor, umbral, origen y todos los sensores en ese instante. Permite queries por máquina.
- **Tabla `lecturas`**: todas las lecturas, con clave primaria compuesta `(maquina_id, timestamp)`. Los datos se ordenan de más reciente a más antiguo.

#### InfluxDB (series temporales)

InfluxDB es una base de datos especializada en series temporales, optimizada para escritura y consulta de datos con timestamp. Almacena todas las lecturas como *measurements* `sensor_data` con tags `maquina_id` y `es_anomalia`, y fields para cada variable. Grafana se conecta directamente a InfluxDB mediante el lenguaje de consulta Flux.

#### MinIO (almacenamiento distribuido S3-compatible)

MinIO es un servidor de almacenamiento de objetos de alto rendimiento, completamente compatible con la API de Amazon S3. Actúa como la capa de **almacenamiento histórico distribuido** del sistema, equivalente funcional a Hadoop HDFS o AWS S3 pero desplegable en local con Docker.

El consumer escribe los datos en formato **Apache Parquet** (columnar, comprimido) cada 100 lecturas en dos buckets:

- `smartmanutech-historical/lecturas/`: todas las lecturas de sensores.
- `smartmanutech-anomalias/anomalias/`: solo las lecturas clasificadas como anómalas.

Los archivos Parquet son directamente consultables con Spark, Pandas o cualquier herramienta de análisis compatible con S3. La consola web de MinIO está disponible en `http://localhost:9001`.

**¿Por qué MinIO en lugar de Hadoop HDFS?**

| Característica        | Hadoop HDFS     | MinIO               |
|-----------------------|-----------------|---------------------|
| Compatibilidad S3     | No nativa       | Total (API S3)      |
| Requisitos RAM        | >4 GB extra     | <512 MB             |
| Despliegue Docker     | Complejo        | Un solo contenedor  |
| Formato de datos      | Cualquiera      | Objetos (Parquet)   |
| Uso en industria      | Legacy          | Estándar moderno    |

MinIO es la opción estándar en arquitecturas cloud-native modernas y es el backend utilizado por herramientas como MLflow, Dask y Apache Spark en entornos de producción.



Expone los datos a través de endpoints REST completos:                          | Descripción                              |
|-----------------------------------|------------------------------------------|
| `GET /`                           | Health check básico                      |
| `GET /health`                     | Estado de todas las conexiones           |
| `GET /anomalias/recientes`        | Últimas N anomalías (Cassandra)          |
| `GET /anomalias/maquina/{id}`     | Anomalías de una máquina                 |
| `GET /maquina/{id}/estado`        | Última lectura conocida de la máquina    |
| `GET /maquinas`                   | Lista de máquinas monitorizadas          |
| `GET /metricas/resumen`           | Totales del sistema                      |
| `GET /metricas/serie-temporal`    | Datos históricos de InfluxDB             |

#### Grafana (visualización)

Dashboard con 7 paneles organizados en dos filas:

1. **Fila "Sensores IoT":**
   - Temperatura por máquina (°C) — umbral visual en 85 y 95 °C
   - Vibración (mm/s) — umbral en 3,5 y 4,5 mm/s
   - RPM por máquina
   - Consumo eléctrico (kW)
   - Presión (bar)

2. **Fila "Anomalías":**
   - Contador de anomalías en el rango seleccionado (stat panel con fondo en color)
   - Serie temporal de anomalías agrupadas en ventanas de 30 segundos

El dashboard se refresca automáticamente cada 5 segundos.

### 4.3 Modelo de Machine Learning — PySpark MLlib KMeans

**¿Por qué PySpark MLlib?**

PySpark MLlib es la biblioteca de Machine Learning distribuido de Apache Spark. Permite entrenar modelos sobre datasets de gran volumen aprovechando el procesamiento paralelo del clúster. En este proyecto se utiliza un `SparkSession` local que puede escalar a un clúster real sin cambiar el código.

**Algoritmo: KMeans no supervisado**

El problema de detección de anomalías en este contexto es no supervisado: no se dispone de datos históricos etiquetados como "fallo / no fallo". KMeans agrupa los datos en k=2 clústeres: uno representando el funcionamiento normal y otro las anomalías.

La estrategia de detección:
- Los puntos situados a mayor distancia del centroide de su clúster son considerados anómalos.
- El umbral se fija en el percentil 95 de las distancias calculadas durante el entrenamiento.

**Pipeline de PySpark MLlib:**

```
VectorAssembler → StandardScaler → KMeans(k=2)
```

1. `VectorAssembler`: concatena los 5 campos de sensor en un vector denso.
2. `StandardScaler`: normaliza cada columna (media=0, std=1) para que las variables con distintas unidades tengan el mismo peso.
3. `KMeans(k=2, maxIter=30)`: agrupa los datos; el modelo exporta los centroides.

**Entrenamiento y exportación de parámetros:**

El entrenamiento se realiza con PySpark MLlib. Los parámetros resultantes (centroides, media y std del scaler) se exportan a numpy para la **inferencia en tiempo real** sin overhead de Spark, combinando lo mejor de ambos mundos.

**Datos de entrenamiento:**

Se simulan 2.000 registros de operación normal más un 5% de anomalías inyectadas, con los parámetros estadísticos de las 5 máquinas.

**Reentrenamiento online:**

Cada 500 nuevos registros procesados, el modelo se reentrena con PySpark MLlib usando esos datos (adaptación a *concept drift*).

**Comparación Isolation Forest vs KMeans MLlib:**

| Característica           | Isolation Forest (sklearn) | KMeans (PySpark MLlib)   |
|--------------------------|---------------------------|--------------------------|
| Motor de ejecución       | Single-node               | Distribuido (Spark)      |
| Escala a big data        | Limitado                  | Sí (clúster Spark)       |
| Datos etiquetados        | No requiere               | No requiere              |
| Detecta patrones compl.  | Sí                        | Sí (por distancia)       |
| Integración pipeline     | sklearn Pipeline          | PySpark ML Pipeline      |

---

## 5. Instrucciones de ejecución

### Requisitos previos

- Docker Engine ≥ 24.0
- Docker Compose ≥ 2.20
- 4 GB de RAM disponibles (Cassandra necesita ≥ 2 GB)
- Puertos libres: 9092, 9042, 8086, 3005, 8000

### Paso 1 — Arrancar la plataforma

```bash
cd "/ruta/a/Big Data Aplicado"
docker compose up --build -d
```

Docker construye las imágenes Python (producer, consumer, api) e inicia los 7 servicios.

### Paso 2 — Esperar a que todo esté listo

```bash
docker compose ps
```

Cassandra tarda ~60 s en pasar a `healthy`. Los servicios Python (`restart: on-failure`) reinician automáticamente hasta que Cassandra e InfluxDB estén disponibles.

### Paso 3 — Ver el sistema funcionando

```bash
# Productor enviando 10 datos/s
docker logs -f bigdata-producer

# Consumer procesando y detectando anomalías
docker logs -f bigdata-consumer
```

### Paso 4 — Grafana

Navegar a http://localhost:3005 (usuario: `admin`, contraseña: `admin123`).

El dashboard SmartManuTech se muestra como página de inicio. Las gráficas muestran datos en tiempo real con refresco cada 5 segundos.

### Paso 5 — API REST

```bash
# Documentación interactiva completa
http://localhost:8000/docs

# Ejemplos rápidos en terminal:
curl http://localhost:8000/health
curl http://localhost:8000/anomalias/recientes?limite=5
curl http://localhost:8000/maquina/MAQ-002/estado
```

### Paso 6 — Detener

```bash
docker compose down          # Conserva datos (volúmenes)
docker compose down -v       # Elimina datos también
```

---

## 6. Problemas encontrados y soluciones

### 6.1 Puerto 3000 ocupado en arranque de Grafana

**Problema:** El error `address already in use` al arrancar Grafana indicaba que el puerto 3000 ya estaba en uso por otra aplicación en el host.

**Solución:** Se modificó el mapeo de puertos de `3000:3000` a `3005:3000` en el `docker-compose.yml`. Grafana sigue escuchando en su puerto interno 3000, pero se expone externamente en el 3005.

### 6.2 IP hardcodeada en el productor

**Problema:** El productor original tenía la IP `192.168.219.29:9092` hardcodeada, lo que impedía funcionar en cualquier otra máquina.

**Solución:** Se sustituyó por `kafka:29092` (nombre del servicio Docker dentro de la red `bigdata_net`) leído desde la variable de entorno `KAFKA_BOOTSTRAP_SERVERS`. Esto hace el código portable y configurable sin modificarlo.

### 6.3 Consumer dependía de Cassandra no inicializada

**Problema:** Cassandra tarda ~60 segundos en arrancar completamente. El consumer intentaba conectar antes de que estuviera lista y fallaba.

**Solución:** Se implementó lógica de reintentos con espera exponencial (hasta 12 reintentos con 8 segundos de espera) tanto para Cassandra como para Kafka. Además, en el `docker-compose.yml` se añadió `condition: service_healthy` para que los servicios Python no arranquen hasta que Cassandra pase su healthcheck.

### 6.4 Grafana no encontraba el datasource InfluxDB

**Problema:** El dashboard hacía referencia a un datasource con UID `influxdb-smartmanutech` pero el datasource provisionado no tenía ese UID.

**Solución:** Se añadió explícitamente `uid: influxdb-smartmanutech` en el archivo `grafana-datasource.yml`, garantizando que el UID del dashboard JSON y el UID del datasource coincidan.

### 6.5 Módulo ML no encontrado en el container del consumer

**Problema:** Al ejecutar `consumer_anomalias.py` dentro del container, el import `from ml.predictive_model import DetectorAnomalias` fallaba porque Python no encontraba el paquete `ml`.

**Solución:** En el `Dockerfile` del consumer, se copia el directorio `ml/` al `WORKDIR /app`, y se añade `sys.path.insert(0, '/app')` en el script. Además se creó `ml/__init__.py` para que Python lo reconozca como paquete. 

---

## 7. Evaluación del sistema

### 7.1 Tiempo de respuesta

Se midió desde la generación del dato en el productor hasta su aparición en Grafana:

| Etapa                              | Latencia estimada |
|------------------------------------|-------------------|
| Producer → Kafka (envío)           | < 50 ms           |
| Kafka → Consumer (lectura)         | < 100 ms          |
| Consumer → InfluxDB (escritura)    | < 200 ms          |
| InfluxDB → Grafana (refresh 5s)    | hasta 5.000 ms    |
| **Latencia total extremo a extremo** | **< 5,5 s**     |

En un sistema de producción real con Kafka optimizado y Grafana con refresh de 1 segundo, la latencia perceptible podría reducirse a < 2 segundos.

### 7.2 Escalabilidad

La arquitectura escala horizontalmente en todos sus componentes:

- **Kafka:** Se pueden añadir brokers y particiones en el topic para procesar más mensajes en paralelo.
- **Consumer:** Con particiones Kafka, se pueden lanzar múltiples instancias del consumer en paralelo (consumer group `smartmanutech-group-v1`).
- **Cassandra:** Diseñado para escalar añadiendo nodos al clúster sin downtime.
- **InfluxDB:** La versión Cloud escala automáticamente; la versión Enterprise permite clustering.

En este proyecto académico se usa 1 broker Kafka, 1 nodo Cassandra y 1 instancia de InfluxDB, lo que es suficiente para demostrar la arquitectura. En producción con 500+ máquinas se requerirían mínimo 3 brokers Kafka y un clúster Cassandra de 3 nodos.

### 7.3 Comparación de modelos de detección

| Modelo                     | Tipo          | Requiere etiquetas | Detecta combinaciones | Interpretabilidad | Velocidad |
|----------------------------|---------------|--------------------|-----------------------|-------------------|-----------|
| Umbrales fijos             | Reglas        | No                 | No                    | Alta              | Muy alta  |
| Isolation Forest           | ML no superv. | No                 | Sí                    | Media             | Alta      |
| One-Class SVM              | ML no superv. | No                 | Sí                    | Baja              | Media     |
| LSTM Autoencoder           | Deep Learning | No (algunos)       | Sí                    | Muy baja          | Baja      |
| Random Forest Clasificador | ML superv.    | Sí                 | Sí                    | Alta              | Alta      |

**Sistema elegido: Umbrales + Isolation Forest**

La combinación de umbrales (para detectar violaciones claras de límites operativos) con Isolation Forest (para detectar anomalías multivariantes sutiles) es la más adecuada para este escenario por:

1. No requiere datos históricos etiquetados (no disponibles en SmartManuTech).
2. Bajo costo computacional, ejecutable en tiempo real.
3. Interpretabilidad razonable: el operario entiende "temperatura alta" pero también recibe alertas de patrones inusuales.
4. Extensible: cuando se acumulen suficientes datos etiquetados, se puede añadir un clasificador supervisado en paralelo.

---

## 8. Bibliografía

1. Kreps, J., Narkhede, N., Rao, J. (2011). *Kafka: A distributed messaging system for log processing*. LinkedIn Engineering Blog.

2. Lakshman, A., Malik, P. (2010). *Cassandra: A decentralized structured storage system*. ACM SIGOPS Operating Systems Review.

3. Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). *Isolation Forest*. Proceedings of the 8th IEEE International Conference on Data Mining.

4. Influxdata Inc. (2023). *InfluxDB 2.x Documentation*. https://docs.influxdata.com/influxdb/v2/

5. Ramírez, S. (2023). *FastAPI Documentation*. https://fastapi.tiangolo.com/

6. Confluent Inc. (2023). *Apache Kafka Documentation*. https://kafka.apache.org/documentation/

7. Marz, N., Warren, J. (2015). *Big Data: Principles and best practices of scalable realtime data systems*. Manning Publications.

8. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media.

---

## Anexo A — Estructura del mensaje IoT

Ejemplo de mensaje JSON generado por el productor:

```json
{
  "maquina_id": "MAQ-003",
  "timestamp": "2026-03-24T14:35:22.841503",
  "temperatura": 74.82,
  "vibracion": 2.541,
  "velocidad_rpm": 1492.3,
  "consumo_kw": 44.17,
  "presion_bar": 5.97,
  "error_code": 0,
  "es_anomalia": false
}
```

Ejemplo con anomalía de sobrecalentamiento inyectada:

```json
{
  "maquina_id": "MAQ-002",
  "timestamp": "2026-03-24T14:35:40.123456",
  "temperatura": 118.43,
  "vibracion": 2.71,
  "velocidad_rpm": 1518.0,
  "consumo_kw": 47.92,
  "presion_bar": 5.88,
  "error_code": 1,
  "es_anomalia": true
}
```

## Anexo B — Esquema de Cassandra

```sql
-- Keyspace
CREATE KEYSPACE smartmanutech
  WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

-- Tabla de anomalías detectadas
CREATE TABLE smartmanutech.anomalias (
    id            UUID PRIMARY KEY,
    maquina_id    TEXT,
    timestamp     TIMESTAMP,
    tipo_anomalia TEXT,    -- UMBRAL_ALTO_TEMPERATURA, ANOMALIA_ML, etc.
    campo         TEXT,
    valor         DOUBLE,
    umbral        DOUBLE,
    origen        TEXT,    -- 'umbral' o 'isolation_forest'
    temperatura   DOUBLE,
    vibracion     DOUBLE,
    velocidad_rpm DOUBLE,
    consumo_kw    DOUBLE,
    presion_bar   DOUBLE,
    error_code    INT
);

-- Tabla de histórico completo de lecturas
CREATE TABLE smartmanutech.lecturas (
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
) WITH CLUSTERING ORDER BY (timestamp DESC);
```

## Anexo C — Endpoints de la API (referencia rápida)

```
GET  /                                    → Health check
GET  /health                              → Estado de Cassandra e InfluxDB
GET  /maquinas                            → Lista de 5 máquinas
GET  /maquina/{maquina_id}/estado        → Última lectura de una máquina
GET  /anomalias/recientes?limite=20      → Últimas anomalías (todas las máquinas)
GET  /anomalias/maquina/{maquina_id}     → Anomalías de una máquina concreta
GET  /metricas/resumen                   → Contadores globales
GET  /metricas/serie-temporal            → Datos InfluxDB con parámetros:
       ?campo=temperatura                    (temperatura|vibracion|velocidad_rpm|consumo_kw|presion_bar)
       &maquina_id=MAQ-001                  (opcional)
       &rango=-10m                          (-5m|-1h|-24h)
```

Swagger UI completo en: **http://localhost:8000/docs**
