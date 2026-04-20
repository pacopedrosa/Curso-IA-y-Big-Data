# EduRecom — Sistema de Recomendación de Cursos

Sistema de recomendación educativa basado en **aprendizaje no supervisado** (K-Means + DBSCAN)
y **filtrado colaborativo** (User-Based CF), con API REST y visualización en tiempo real.

## Estructura del proyecto

```
recommendation_system/
├── data/
│   └── generate_data.py       # Generación de datos sintéticos
├── models/
│   ├── clustering.py          # K-Means + DBSCAN + visualizaciones PCA
│   └── collaborative.py       # Filtrado colaborativo + CollaborativeFilteringModel + Precision@K
├── database/
│   ├── init_db.py             # Inicialización completa (datos + clustering)
│   └── edtech.db              # SQLite (generado al ejecutar init_db.py)
├── api/
│   └── main.py                # API FastAPI v1.0 (REST)
├── app/
│   ├── streamlit_app.py       # Interfaz visual (Streamlit)
│   └── static/                # Gráficos PNG generados
├── requirements.txt
├── run.py                     # Arranque todo-en-uno
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Opción A — Todo-en-uno (recomendado)
```bash
python run.py
```

### Opción B — Paso a paso

**1. Inicializar base de datos y entrenar modelos:**
```bash
python database/init_db.py
```

**2. Arrancar la API FastAPI:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Documentación interactiva: http://localhost:8000/docs
```

**3. Arrancar la interfaz Streamlit:**
```bash
streamlit run app/streamlit_app.py
# Interfaz: http://localhost:8501
```

## Endpoints API principales

| Método | Ruta | Parámetros | Descripción |
|--------|------|------------|-------------|
| GET | `/` | — | Health check |
| GET | `/users` | `limit`, `offset` | Lista paginada de usuarios |
| GET | `/users/{id}` | — | Detalle de un usuario |
| GET | `/users/{id}/recommendations` | `top_n` (1-20, defecto 5) | Recomendaciones personalizadas |
| GET | `/users/{id}/cluster` | — | Cluster asignado al usuario |
| GET | `/courses` | `category`, `level` | Catálogo de cursos (filtros opcionales) |
| GET | `/clusters/summary` | — | Resumen estadístico por cluster |
| GET | `/metrics` | `sample_size` (10-200, defecto 50) | Silhouette score + Precision@K |
| GET | `/docs` | — | Swagger UI interactivo |

## Detalles técnicos

- **Dataset**: 500 usuarios, 50 cursos, ~4 500 interacciones sintéticas
- **Clustering**: K-Means (k=5, seleccionado por índice de Silhouette); DBSCAN para detección de outliers; visualización con PCA 2D
- **Recomendación**: `CollaborativeFilteringModel` — similitud coseno entre usuarios del mismo cluster; predicción por media ponderada de vecinos; fallback global si el cluster tiene pocos usuarios
- **Métricas**: Índice de Silhouette (clustering) · Precision@K con holdout multi-ítem del 25% (mínimo 1, máximo 5 ítems retirados por usuario)
- **Base de datos**: SQLite embebida (sin dependencias externas)
- **Privacidad**: datos completamente sintéticos, sin PII
