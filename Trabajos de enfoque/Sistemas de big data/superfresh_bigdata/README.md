# SuperFresh Sales Intelligence 🛒

Sistema de predicción de demanda basado en Big Data y Machine Learning para la cadena de supermercados **SuperFresh**.

## Estructura del proyecto

```
superfresh_bigdata/
├── data/
│   ├── generate_data.py       # Generación de datos históricos sintéticos (3 años)
│   └── spark_processing.py    # Procesamiento Big Data con Apache Spark
├── models/
│   └── prediction.py          # ARIMA, Random Forest, Gradient Boosting
├── api/
│   └── main.py                # API REST con FastAPI
├── app/
│   ├── streamlit_app.py       # Dashboard / Cuadro de mandos
│   └── static/                # Gráficos generados
├── database/
│   ├── init_db.py             # Inicialización de BD y entrenamiento
│   ├── storage.py             # Soporte PostgreSQL y MongoDB
│   └── superfresh.db          # SQLite (se crea automáticamente)
├── requirements.txt
└── run.py                     # Script de arranque
```

## Datos simulados

- **3 años** de ventas diarias (2023–2025)
- **25 productos** en 10 categorías
- **5 tiendas** en Madrid, Sevilla, Valencia y Alicante
- Variables: estacionalidad, eventos especiales, promociones, clima, día de la semana

## Modelos implementados

| Modelo | Descripción |
|---|---|
| **Random Forest** | Ensemble de árboles con 200 estimadores, variables de lag y clima |
| **Gradient Boosting** | Boosting con 150 estimadores, ideal para patrones no lineales |
| **ARIMA(2,d,2)** | Modelo de series temporales, test ADF automático |

Métricas de evaluación: **MAE**, **RMSE**, **R²**, **MAPE**

### Optimización de hiperparámetros

El pipeline soporta ajuste automático con `RandomizedSearchCV` y validación cruzada temporal (`TimeSeriesSplit`):

```bash
python models/prediction.py --tune
```

## Arquitectura Big Data

| Capa | Tecnología |
|---|---|
| **Procesamiento distribuido** | Apache Spark (PySpark) |
| **Almacenamiento dev** | SQLite |
| **Almacenamiento producción** | PostgreSQL · MongoDB |
| **API REST** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit + Plotly |

## Instalación y ejecución

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Inicializar base de datos y entrenar modelos (~2 min)
python database/init_db.py

# 4. Arrancar la aplicación
python run.py
```

- **API REST**: http://localhost:8002
- **Documentación API**: http://localhost:8002/docs
- **Dashboard**: http://localhost:8502

### Procesamiento con Apache Spark (opcional)

```bash
# Requiere Java 8+ instalado
python data/spark_processing.py
```

### Exportar datos a PostgreSQL o MongoDB

```bash
# PostgreSQL
python database/storage.py postgresql --host localhost --user postgres --password secret

# MongoDB
python database/storage.py mongodb --uri mongodb://localhost:27017
```

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Health check |
| GET | `/products` | Catálogo de productos |
| GET | `/stores` | Tiendas disponibles |
| GET | `/categories` | Categorías disponibles |
| GET | `/sales/history` | Historial de ventas con filtros |
| GET | `/sales/monthly-trend` | Tendencia mensual |
| GET | `/sales/top-products` | Ranking de productos |
| GET | `/sales/seasonal` | Análisis de estacionalidad |
| POST | `/predict` | Predicción puntual (RF / GB) |
| POST | `/predict/batch` | Previsión multi-día |
| GET | `/metrics` | Métricas de los modelos |
| GET | `/metrics/all` | Métricas + hiperparámetros óptimos |
