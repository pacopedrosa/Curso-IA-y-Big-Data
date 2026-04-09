# SuperFresh Sales Intelligence 🛒

Sistema de predicción de demanda basado en Big Data y Machine Learning para la cadena de supermercados **SuperFresh**.

## Estructura del proyecto

```
superfresh_bigdata/
├── data/
│   └── generate_data.py       # Generación de datos históricos sintéticos (3 años)
├── models/
│   └── prediction.py          # ARIMA, Random Forest, Gradient Boosting
├── api/
│   └── main.py                # API REST con FastAPI
├── app/
│   ├── streamlit_app.py       # Dashboard / Cuadro de mandos
│   └── static/                # Gráficos generados
├── database/
│   ├── init_db.py             # Inicialización de BD y entrenamiento
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

- **API REST**: http://localhost:8001
- **Documentación API**: http://localhost:8001/docs
- **Dashboard**: http://localhost:8501

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Health check |
| GET | `/products` | Catálogo de productos |
| GET | `/stores` | Tiendas disponibles |
| GET | `/sales/history` | Historial de ventas con filtros |
| GET | `/sales/monthly-trend` | Tendencia mensual |
| GET | `/sales/top-products` | Ranking de productos |
| POST | `/predict` | Predicción puntual |
| POST | `/predict/batch` | Previsión multi-día |
| GET | `/metrics` | Métricas de los modelos |
| GET | `/categories` | Categorías disponibles |
