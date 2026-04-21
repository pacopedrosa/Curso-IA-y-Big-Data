"""
API REST con FastAPI para consultar predicciones de ventas de SuperFresh.
Endpoints para predicción, historial, productos y métricas del modelo.
"""
from __future__ import annotations
import os
import sys
import pickle
import sqlite3
import warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# Añadir raíz del proyecto al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DB_PATH       = os.path.join(ROOT, "database", "superfresh.db")
MODEL_DIR     = os.path.join(ROOT, "database")

# ── Estado global de la app ───────────────────────────────────────────────────
_state: dict = {}


def _load_artifacts():
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        _state["rf"] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "gb_model.pkl"), "rb") as f:
        _state["gb"] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
        _state["feature_cols"] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "rb") as f:
        _state["metrics"] = pickle.load(f)

    conn = sqlite3.connect(DB_PATH)
    _state["sales"]    = pd.read_sql("SELECT * FROM sales", conn, parse_dates=["sale_date"])
    _state["products"] = pd.read_sql("SELECT * FROM products", conn)
    _state["stores"]   = pd.read_sql("SELECT * FROM stores", conn)
    _state["weather"]  = pd.read_sql("SELECT * FROM weather", conn)
    conn.close()
    print("[API] Artefactos cargados correctamente.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield
    _state.clear()


app = FastAPI(
    title="SuperFresh Sales Prediction API",
    description="API para predicción de demanda de productos en SuperFresh",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos Pydantic ──────────────────────────────────────────────────────────
class ProductOut(BaseModel):
    product_id: int
    name: str
    category: str
    base_price: float
    base_demand: int


class StoreOut(BaseModel):
    store_id: int
    name: str
    city: str
    store_type: str


class SalesSummary(BaseModel):
    date: str
    store_id: int
    product_id: int
    units_sold: int
    revenue: float


class PredictionRequest(BaseModel):
    product_id: int = Field(..., example=1)
    store_id: int   = Field(..., example=1)
    target_date: str = Field(..., example="2026-01-15",
                             description="Fecha a predecir (YYYY-MM-DD)")
    discount_pct: float = Field(0.0, ge=0.0, le=100.0, description="% de descuento")
    model: str = Field("random_forest", description="random_forest | gradient_boosting")


class PredictionOut(BaseModel):
    product_id: int
    product_name: str
    store_id: int
    store_name: str
    target_date: str
    predicted_units: float
    predicted_revenue: float
    model_used: str
    confidence_lower: float
    confidence_upper: float


class BatchPredictionRequest(BaseModel):
    product_id: int
    store_id: int
    days_ahead: int = Field(30, ge=1, le=365)
    discount_pct: float = Field(0.0, ge=0.0, le=100.0)
    model: str = "random_forest"


class MetricsOut(BaseModel):
    random_forest: dict
    gradient_boosting: dict


class TopProductOut(BaseModel):
    product_id: int
    name: str
    category: str
    total_units: int
    total_revenue: float


class MonthlyTrend(BaseModel):
    year_month: str
    total_revenue: float
    total_units: int


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_product(product_id: int) -> pd.Series:
    row = _state["products"][_state["products"]["product_id"] == product_id]
    if row.empty:
        raise HTTPException(404, f"Producto {product_id} no encontrado")
    return row.iloc[0]


def _get_store(store_id: int) -> pd.Series:
    row = _state["stores"][_state["stores"]["store_id"] == store_id]
    if row.empty:
        raise HTTPException(404, f"Tienda {store_id} no encontrada")
    return row.iloc[0]


def _build_features_for_date(product_id: int, store_id: int,
                               target: date, discount_pct: float) -> np.ndarray:
    sales = _state["sales"]
    product = _get_product(product_id)
    unit_price = round(product["base_price"] * (1 - discount_pct / 100), 4)

    # Histórico filtrado
    hist = (sales[(sales["product_id"] == product_id) & (sales["store_id"] == store_id)]
            .sort_values("sale_date"))

    target_ts = pd.Timestamp(target)
    before = hist[hist["sale_date"] < target_ts]["units_sold"]

    def safe_lag(n: int) -> float:
        return float(before.iloc[-n]) if len(before) >= n else float(before.mean() or 0)

    def safe_roll(n: int) -> float:
        return float(before.iloc[-n:].mean()) if len(before) >= n else float(before.mean() or 0)

    # Clima
    weather = _state["weather"]
    w_row = weather[(weather["store_id"] == store_id) &
                    (weather["year"] == target.year) &
                    (weather["month"] == target.month)]
    avg_temp = float(w_row["avg_temp_c"].values[0]) if not w_row.empty else 15.0
    rain_mm  = float(w_row["rain_mm"].values[0]) if not w_row.empty else 30.0

    feats = {
        "product_id":   product_id,
        "store_id":     store_id,
        "year":         target.year,
        "month":        target.month,
        "day":          target.day,
        "day_of_week":  target.weekday(),
        "week_of_year": target.isocalendar()[1],
        "is_weekend":   int(target.weekday() >= 5),
        "quarter":      (target.month - 1) // 3 + 1,
        "unit_price":   unit_price,
        "discount_pct": discount_pct,
        "lag_7":        safe_lag(7),
        "lag_14":       safe_lag(14),
        "lag_30":       safe_lag(30),
        "roll_7":       safe_roll(7),
        "roll_30":      safe_roll(30),
        "avg_temp_c":   avg_temp,
        "rain_mm":      rain_mm,
    }
    return np.array([[feats[c] for c in _state["feature_cols"]]])


def _predict(model_name: str, X: np.ndarray) -> float:
    key = "rf" if model_name == "random_forest" else "gb"
    model = _state[key]
    return max(0.0, float(model.predict(X)[0]))


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "SuperFresh Prediction API v1.0.0"}


@app.get("/products", response_model=list[ProductOut], tags=["Catálogo"])
def list_products(category: Optional[str] = None):
    df = _state["products"]
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    return df.to_dict(orient="records")


@app.get("/products/{product_id}", response_model=ProductOut, tags=["Catálogo"])
def get_product(product_id: int):
    return _get_product(product_id).to_dict()


@app.get("/stores", response_model=list[StoreOut], tags=["Catálogo"])
def list_stores():
    return _state["stores"].to_dict(orient="records")


@app.get("/sales/history", tags=["Historial"])
def sales_history(
    product_id: Optional[int]  = None,
    store_id:   Optional[int]  = None,
    start_date: Optional[str]  = Query(None, example="2025-01-01"),
    end_date:   Optional[str]  = Query(None, example="2025-12-31"),
    limit: int = Query(200, ge=1, le=5000),
):
    df = _state["sales"].copy()
    if product_id: df = df[df["product_id"] == product_id]
    if store_id:   df = df[df["store_id"]   == store_id]
    if start_date: df = df[df["sale_date"] >= pd.Timestamp(start_date)]
    if end_date:   df = df[df["sale_date"] <= pd.Timestamp(end_date)]
    df = df.sort_values("sale_date", ascending=False).head(limit)
    df["sale_date"] = df["sale_date"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.get("/sales/monthly-trend", response_model=list[MonthlyTrend], tags=["Análisis"])
def monthly_trend(store_id: Optional[int] = None):
    df = _state["sales"].copy()
    if store_id:
        df = df[df["store_id"] == store_id]
    df["year_month"] = df["sale_date"].dt.to_period("M").astype(str)
    grouped = df.groupby("year_month").agg(
        total_revenue=("revenue", "sum"),
        total_units=("units_sold", "sum"),
    ).reset_index()
    return grouped.to_dict(orient="records")


@app.get("/sales/top-products", response_model=list[TopProductOut], tags=["Análisis"])
def top_products(n: int = Query(10, ge=1, le=25),
                 store_id: Optional[int] = None):
    df = _state["sales"].copy()
    if store_id:
        df = df[df["store_id"] == store_id]
    agg = df.groupby("product_id").agg(
        total_units=("units_sold", "sum"),
        total_revenue=("revenue", "sum"),
    ).reset_index()
    agg = agg.merge(_state["products"][["product_id", "name", "category"]], on="product_id")
    agg = agg.sort_values("total_revenue", ascending=False).head(n)
    return agg.to_dict(orient="records")


@app.post("/predict", response_model=PredictionOut, tags=["Predicción"])
def predict_single(req: PredictionRequest):
    target = date.fromisoformat(req.target_date)
    product = _get_product(req.product_id)
    store   = _get_store(req.store_id)

    X = _build_features_for_date(req.product_id, req.store_id, target, req.discount_pct)
    pred = _predict(req.model, X)

    unit_price = round(float(product["base_price"]) * (1 - req.discount_pct / 100), 4)
    revenue    = round(pred * unit_price, 2)

    # Intervalo de confianza aproximado (±15%)
    lower = round(max(0.0, pred * 0.85), 2)
    upper = round(pred * 1.15, 2)

    return PredictionOut(
        product_id       = req.product_id,
        product_name     = product["name"],
        store_id         = req.store_id,
        store_name       = store["name"],
        target_date      = req.target_date,
        predicted_units  = round(pred, 2),
        predicted_revenue= revenue,
        model_used       = req.model,
        confidence_lower = lower,
        confidence_upper = upper,
    )


@app.post("/predict/batch", tags=["Predicción"])
def predict_batch(req: BatchPredictionRequest):
    product = _get_product(req.product_id)
    store   = _get_store(req.store_id)
    unit_price = round(float(product["base_price"]) * (1 - req.discount_pct / 100), 4)

    results = []
    base = date.today()
    for i in range(req.days_ahead):
        target = base + timedelta(days=i)
        X = _build_features_for_date(req.product_id, req.store_id, target, req.discount_pct)
        pred = _predict(req.model, X)
        results.append({
            "date":             target.isoformat(),
            "predicted_units":  round(pred, 2),
            "predicted_revenue":round(pred * unit_price, 2),
        })

    return {
        "product_id":   req.product_id,
        "product_name": product["name"],
        "store_id":     req.store_id,
        "store_name":   store["name"],
        "model_used":   req.model,
        "forecast":     results,
    }


@app.get("/metrics", response_model=MetricsOut, tags=["Modelos"])
def get_metrics():
    return _state["metrics"]


@app.get("/metrics/all", tags=["Modelos"])
def get_all_metrics():
    """
    Devuelve las métricas de todos los modelos entrenados:
    Random Forest, Gradient Boosting e hiperparámetros óptimos del ajuste (si se ejecutó con --tune).
    """
    return _state["metrics"]


@app.get("/categories", tags=["Catálogo"])
def list_categories():
    cats = _state["products"]["category"].unique().tolist()
    return {"categories": sorted(cats)}


@app.get("/sales/seasonal", tags=["Análisis"])
def seasonal_analysis(
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    store_id: Optional[int] = Query(None, description="Filtrar por tienda"),
):
    """
    Calcula las unidades medias por mes para analizar la estacionalidad
    de la demanda, con filtros opcionales por categoría y tienda.
    """
    sales    = _state["sales"].copy()
    products = _state["products"].copy()

    df = sales.merge(products[["product_id", "category"]], on="product_id")
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if store_id:
        df = df[df["store_id"] == store_id]

    df["month"] = df["sale_date"].dt.month
    result = (df.groupby("month")
              .agg(avg_units=("units_sold", "mean"),
                   total_units=("units_sold", "sum"))
              .round(2)
              .reset_index())
    return result.to_dict(orient="records")

