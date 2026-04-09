"""
Modelos de predicción de ventas para SuperFresh.
Implementa ARIMA, Prophet (si disponible) y Random Forest.
Evalúa con MAE, RMSE y R².
"""
import os
import sqlite3
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "superfresh.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "database")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "static")


# ── Carga de datos ─────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(DB_PATH)
    sales = pd.read_sql("SELECT * FROM sales", conn, parse_dates=["sale_date"])
    products = pd.read_sql("SELECT * FROM products", conn)
    stores = pd.read_sql("SELECT * FROM stores", conn)
    weather = pd.read_sql("SELECT * FROM weather", conn)
    conn.close()
    return sales, products, stores, weather


# ── Ingeniería de características ─────────────────────────────────────────────
def build_features(sales: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = sales.copy()
    df["year"] = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month
    df["day"] = df["sale_date"].dt.day
    df["day_of_week"] = df["sale_date"].dt.dayofweek  # 0=Lunes
    df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df["sale_date"].dt.quarter

    # Lags y medias móviles por producto+tienda
    df = df.sort_values(["store_id", "product_id", "sale_date"])
    key = ["store_id", "product_id"]
    df["lag_7"]   = df.groupby(key)["units_sold"].shift(7)
    df["lag_14"]  = df.groupby(key)["units_sold"].shift(14)
    df["lag_30"]  = df.groupby(key)["units_sold"].shift(30)
    df["roll_7"]  = df.groupby(key)["units_sold"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["roll_30"] = df.groupby(key)["units_sold"].transform(lambda x: x.shift(1).rolling(30).mean())

    # Unir clima
    df = df.merge(weather[["store_id", "year", "month", "avg_temp_c", "rain_mm"]],
                  on=["store_id", "year", "month"], how="left")
    df = df.dropna(subset=["lag_7", "lag_14", "lag_30", "roll_7", "roll_30"])
    return df


# ── Métricas ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4), "MAPE": round(mape, 2)}


# ── Random Forest ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "product_id", "store_id", "year", "month", "day", "day_of_week",
    "week_of_year", "is_weekend", "quarter", "unit_price", "discount_pct",
    "lag_7", "lag_14", "lag_30", "roll_7", "roll_30",
    "avg_temp_c", "rain_mm",
]

def train_random_forest(df: pd.DataFrame) -> tuple[RandomForestRegressor, dict]:
    df = df[FEATURE_COLS + ["units_sold", "sale_date"]].dropna()
    df = df.sort_values("sale_date")

    split_date = pd.Timestamp("2025-07-01")
    train = df[df["sale_date"] < split_date]
    test  = df[df["sale_date"] >= split_date]

    X_train = train[FEATURE_COLS]
    y_train = train["units_sold"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["units_sold"]

    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    metrics = compute_metrics(y_test.values, np.maximum(y_pred, 0))
    print(f"[RF] Métricas test: {metrics}")
    return rf, metrics


def train_gradient_boosting(df: pd.DataFrame) -> tuple[GradientBoostingRegressor, dict]:
    df = df[FEATURE_COLS + ["units_sold", "sale_date"]].dropna()
    df = df.sort_values("sale_date")

    split_date = pd.Timestamp("2025-07-01")
    train = df[df["sale_date"] < split_date]
    test  = df[df["sale_date"] >= split_date]

    X_train = train[FEATURE_COLS]
    y_train = train["units_sold"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["units_sold"]

    gb = GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                   learning_rate=0.08, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    metrics = compute_metrics(y_test.values, np.maximum(y_pred, 0))
    print(f"[GB] Métricas test: {metrics}")
    return gb, metrics


# ── ARIMA por producto (serie temporal agregada) ──────────────────────────────
def _adf_test(series: pd.Series) -> bool:
    """True si la serie es estacionaria."""
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1] < 0.05


def train_arima_for_product(sales: pd.DataFrame, product_id: int,
                             store_id: int = 1) -> tuple[dict, pd.DataFrame]:
    ts = (sales[(sales["product_id"] == product_id) & (sales["store_id"] == store_id)]
          .set_index("sale_date")["units_sold"]
          .resample("W").sum()
          .asfreq("W"))

    train = ts.iloc[:-8]
    test  = ts.iloc[-8:]

    # Elegir d en función de estacionariedad
    d = 0 if _adf_test(train) else 1

    model = ARIMA(train, order=(2, d, 2))
    fitted = model.fit()
    forecast_res = fitted.get_forecast(steps=8)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    metrics = compute_metrics(test.values, np.maximum(np.array(forecast), 0))

    # Predicción futura (8 semanas)
    future_model = ARIMA(ts, order=(2, d, 2)).fit()
    future_fc = future_model.get_forecast(steps=8)
    future_pred = future_fc.predicted_mean
    future_ci   = future_fc.conf_int()

    future_df = pd.DataFrame({
        "date": future_pred.index,
        "forecast": np.maximum(future_pred.values, 0),
        "lower": np.maximum(future_ci.iloc[:, 0].values, 0),
        "upper": future_ci.iloc[:, 1].values,
    })

    return metrics, future_df


# ── Gráficos ──────────────────────────────────────────────────────────────────
def plot_feature_importance(rf: RandomForestRegressor, top_n: int = 15):
    os.makedirs(STATIC_DIR, exist_ok=True)
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.sort_values().plot.barh(ax=ax, color="#2ecc71")
    ax.set_title("Importancia de características — Random Forest", fontsize=13)
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    path = os.path.join(STATIC_DIR, "feature_importance.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Guardado {path}")


def plot_model_comparison(metrics_rf: dict, metrics_gb: dict):
    os.makedirs(STATIC_DIR, exist_ok=True)
    labels = ["MAE", "RMSE", "R²", "MAPE"]
    vals_rf = [metrics_rf["MAE"], metrics_rf["RMSE"], metrics_rf["R2"], metrics_rf["MAPE"]]
    vals_gb = [metrics_gb["MAE"], metrics_gb["RMSE"], metrics_gb["R2"], metrics_gb["MAPE"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, vals_rf, width, label="Random Forest", color="#3498db")
    ax.bar(x + width/2, vals_gb, width, label="Gradient Boosting", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Comparación de modelos — Métricas de evaluación", fontsize=13)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(STATIC_DIR, "model_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Guardado {path}")


def plot_arima_forecast(sales: pd.DataFrame, product_id: int, product_name: str,
                        store_id: int = 1):
    os.makedirs(STATIC_DIR, exist_ok=True)
    ts = (sales[(sales["product_id"] == product_id) & (sales["store_id"] == store_id)]
          .set_index("sale_date")["units_sold"]
          .resample("W").sum()
          .asfreq("W"))

    d = 0 if _adf_test(ts) else 1
    fitted = ARIMA(ts, order=(2, d, 2)).fit()
    fc_res = fitted.get_forecast(steps=12)
    fc = fc_res.predicted_mean
    ci = fc_res.conf_int()

    fig, ax = plt.subplots(figsize=(10, 5))
    ts.plot(ax=ax, label="Histórico", color="#2c3e50")
    fc.plot(ax=ax, label="Predicción ARIMA", color="#e74c3c", linestyle="--")
    ax.fill_between(fc.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.3, color="#e74c3c", label="IC 95%")
    ax.set_title(f"Predicción ARIMA — {product_name} (Tienda {store_id})", fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades vendidas (semanal)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(STATIC_DIR, f"arima_product_{product_id}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Guardado {path}")


def plot_monthly_sales_trend(sales: pd.DataFrame):
    os.makedirs(STATIC_DIR, exist_ok=True)
    monthly = (sales.groupby(sales["sale_date"].dt.to_period("M"))["revenue"]
               .sum().reset_index())
    monthly["sale_date"] = monthly["sale_date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(monthly["sale_date"], monthly["revenue"], alpha=0.4, color="#3498db")
    ax.plot(monthly["sale_date"], monthly["revenue"], color="#2980b9", linewidth=2)
    ax.set_title("Evolución de ingresos mensuales — SuperFresh", fontsize=13)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Ingresos (€)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(STATIC_DIR, "monthly_trend.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Guardado {path}")


# ── Guardado/carga de modelos ─────────────────────────────────────────────────
def save_models(rf, gb, metrics_rf, metrics_gb):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(MODEL_DIR, "gb_model.pkl"), "wb") as f:
        pickle.dump(gb, f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(FEATURE_COLS, f)
    # Guardar métricas
    metrics_all = {"random_forest": metrics_rf, "gradient_boosting": metrics_gb}
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics_all, f)
    print("[models] Modelos guardados.")


def load_models():
    rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    gb_path = os.path.join(MODEL_DIR, "gb_model.pkl")
    if not os.path.exists(rf_path):
        raise FileNotFoundError("Modelos no encontrados. Ejecuta init_db.py primero.")
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)
    with open(gb_path, "rb") as f:
        gb = pickle.load(f)
    return rf, gb


# ── Pipeline completo ─────────────────────────────────────────────────────────
def run_model_pipeline():
    print("[pipeline] Cargando datos…")
    sales, products, stores, weather = load_data()

    print("[pipeline] Construyendo características…")
    df_feat = build_features(sales, weather)

    print("[pipeline] Entrenando Random Forest…")
    rf, metrics_rf = train_random_forest(df_feat)

    print("[pipeline] Entrenando Gradient Boosting…")
    gb, metrics_gb = train_gradient_boosting(df_feat)

    print("[pipeline] Guardando modelos…")
    save_models(rf, gb, metrics_rf, metrics_gb)

    print("[pipeline] Generando gráficos…")
    plot_feature_importance(rf)
    plot_model_comparison(metrics_rf, metrics_gb)
    plot_monthly_sales_trend(sales)

    # ARIMA para 3 productos representativos
    for pid in [1, 16, 13]:
        pname = products[products["product_id"] == pid]["name"].values[0]
        try:
            _, _ = train_arima_for_product(sales, pid, store_id=1)
            plot_arima_forecast(sales, pid, pname, store_id=1)
        except Exception as e:
            print(f"[ARIMA] Error en producto {pid}: {e}")

    return metrics_rf, metrics_gb


if __name__ == "__main__":
    run_model_pipeline()
