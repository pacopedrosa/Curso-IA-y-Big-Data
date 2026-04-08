"""
API REST del sistema de recomendación de cursos.
Construida con FastAPI.

Endpoints:
  GET /users                        → lista de usuarios
  GET /users/{user_id}              → detalle de un usuario
  GET /users/{user_id}/cluster      → cluster asignado al usuario
  GET /users/{user_id}/recommendations → recomendaciones top-N
  GET /courses                      → catálogo de cursos
  GET /clusters/summary             → resumen estadístico de cada cluster
  GET /metrics                      → métricas del sistema (Silhouette, Precision@K)
"""
import os
import sys

# Asegurar que el paquete raíz está en el path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import sqlite3
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.collaborative import CollaborativeFilteringModel
from models.clustering import load_data

DB_PATH = os.path.join(ROOT, "database", "edtech.db")


# ── Modelos de respuesta Pydantic ─────────────────────────────────────────────

class UserOut(BaseModel):
    user_id: str
    age: int
    profile: str
    seniority: str


class CourseOut(BaseModel):
    course_id: str
    name: str
    category: str
    level: str
    avg_rating: float
    duration_h: float


class RecommendationOut(BaseModel):
    course_id: str
    name: str
    category: str
    level: str
    predicted_rating: float
    avg_rating: float


class ClusterInfo(BaseModel):
    user_id: str
    kmeans_cluster: int
    dbscan_cluster: int


class ClusterSummary(BaseModel):
    kmeans_cluster: int
    n_users: int
    avg_rating: float
    avg_progress: float
    top_category: str


class MetricsOut(BaseModel):
    precision_at_5: float
    precision_std: float
    n_evaluated: int
    km_silhouette: Optional[float] = None


# ── Estado global de la aplicación ───────────────────────────────────────────

class AppState:
    cf_model: CollaborativeFilteringModel = None
    km_silhouette: float = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga única del modelo al arrancar
    print("Cargando modelo de filtrado colaborativo...")
    state.cf_model = CollaborativeFilteringModel(DB_PATH)
    state.cf_model.fit()

    # Leer silhouette almacenado en BD si existe
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT km_silhouette FROM model_metrics LIMIT 1").fetchone()
        if row:
            state.km_silhouette = row[0]
        conn.close()
    except Exception:
        pass

    print("Sistema listo.")
    yield
    # Limpieza al cerrar (si fuera necesario)


# ── Aplicación ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EduRecom API",
    description="Sistema de recomendación de cursos educativos basado en ML",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Utilidad conexión BD ──────────────────────────────────────────────────────

def get_conn():
    return sqlite3.connect(DB_PATH)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "EduRecom API v1.0"}


@app.get("/users", response_model=list[UserOut], tags=["usuarios"])
def list_users(limit: int = Query(50, ge=1, le=500),
               offset: int = Query(0, ge=0)):
    conn = get_conn()
    rows = conn.execute(
        "SELECT user_id, age, profile, seniority FROM users LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return [UserOut(user_id=r[0], age=r[1], profile=r[2], seniority=r[3]) for r in rows]


@app.get("/users/{user_id}", response_model=UserOut, tags=["usuarios"])
def get_user(user_id: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT user_id, age, profile, seniority FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Usuario '{user_id}' no encontrado")
    return UserOut(user_id=row[0], age=row[1], profile=row[2], seniority=row[3])


@app.get("/users/{user_id}/cluster", response_model=ClusterInfo, tags=["clustering"])
def get_user_cluster(user_id: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT user_id, kmeans_cluster, dbscan_cluster FROM user_clusters WHERE user_id = ?",
        (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Datos de cluster no disponibles para este usuario")
    return ClusterInfo(user_id=row[0], kmeans_cluster=row[1], dbscan_cluster=row[2])


@app.get("/users/{user_id}/recommendations",
         response_model=list[RecommendationOut], tags=["recomendaciones"])
def get_recommendations(user_id: str, top_n: int = Query(5, ge=1, le=20)):
    if state.cf_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    if user_id not in state.cf_model.matrix.index:
        raise HTTPException(status_code=404, detail=f"Usuario '{user_id}' no encontrado en la matriz")

    recs = state.cf_model.recommend(user_id, top_n=top_n)
    if recs.empty:
        return []

    return [
        RecommendationOut(
            course_id=row["course_id"],
            name=row["name"],
            category=row["category"],
            level=row["level"],
            predicted_rating=float(row["predicted_rating"]),
            avg_rating=float(row["avg_rating"]),
        )
        for _, row in recs.iterrows()
    ]


@app.get("/courses", response_model=list[CourseOut], tags=["cursos"])
def list_courses(category: Optional[str] = None, level: Optional[str] = None):
    conn = get_conn()
    query = "SELECT course_id, name, category, level, avg_rating, duration_h FROM courses WHERE 1=1"
    params: list = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if level:
        query += " AND level = ?"
        params.append(level)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [
        CourseOut(course_id=r[0], name=r[1], category=r[2],
                  level=r[3], avg_rating=r[4], duration_h=r[5])
        for r in rows
    ]


@app.get("/clusters/summary", response_model=list[ClusterSummary], tags=["clustering"])
def clusters_summary():
    conn = get_conn()
    df_clusters = pd.read_sql("SELECT * FROM user_clusters", conn)
    df_interactions = pd.read_sql(
        "SELECT i.user_id, i.rating, i.progress_pct, c.category "
        "FROM interactions i JOIN courses c ON i.course_id = c.course_id", conn
    )
    conn.close()

    merged = df_clusters.merge(df_interactions, on="user_id")
    grp = merged.groupby("kmeans_cluster")

    result = []
    for clu, g in grp:
        top_cat = g["category"].value_counts().idxmax()
        result.append(ClusterSummary(
            kmeans_cluster=int(clu),
            n_users=int(g["user_id"].nunique()),
            avg_rating=round(float(g["rating"].mean()), 3),
            avg_progress=round(float(g["progress_pct"].mean()), 2),
            top_category=top_cat,
        ))
    return sorted(result, key=lambda x: x.kmeans_cluster)


@app.get("/metrics", response_model=MetricsOut, tags=["evaluación"])
def get_metrics(sample_size: int = Query(50, ge=10, le=200)):
    if state.cf_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    metrics = state.cf_model.evaluate(sample_size=sample_size)
    return MetricsOut(
        precision_at_5=round(metrics["precision_at_k"], 4),
        precision_std=round(metrics["std"], 4),
        n_evaluated=metrics["n_evaluated"],
        km_silhouette=state.km_silhouette,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
