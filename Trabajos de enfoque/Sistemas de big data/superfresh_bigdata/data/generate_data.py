"""
Generación de datos históricos de ventas sintéticos para SuperFresh.
Simula 3 años de datos diarios con estacionalidad, eventos y promociones.
"""
import random
import sqlite3
import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "superfresh.db")
DATA_DIR = os.path.dirname(__file__)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Catálogo de productos ──────────────────────────────────────────────────────
PRODUCTS = [
    # (id, nombre, categoría, precio_base, demanda_base_diaria)
    (1,  "Leche entera 1L",       "Lácteos",      0.99,  80),
    (2,  "Yogur natural pack-4",  "Lácteos",      1.49,  60),
    (3,  "Queso manchego 250g",   "Lácteos",      3.20,  35),
    (4,  "Pan de molde",          "Panadería",    1.10,  70),
    (5,  "Croissants x6",         "Panadería",    2.10,  40),
    (6,  "Baguette",              "Panadería",    0.80,  55),
    (7,  "Manzana kg",            "Frutas",       1.80,  50),
    (8,  "Naranja kg",            "Frutas",       1.20,  45),
    (9,  "Plátano kg",            "Frutas",       1.50,  65),
    (10, "Tomate kg",             "Verduras",     1.90,  48),
    (11, "Lechuga",               "Verduras",     0.90,  42),
    (12, "Pimiento rojo kg",      "Verduras",     2.30,  30),
    (13, "Pechuga pollo 500g",    "Carnes",       3.50,  55),
    (14, "Carne picada 400g",     "Carnes",       4.20,  45),
    (15, "Lomo de cerdo 500g",    "Carnes",       4.80,  30),
    (16, "Agua mineral 1.5L",     "Bebidas",      0.50, 100),
    (17, "Refresco cola 2L",      "Bebidas",      1.60,  70),
    (18, "Zumo naranja 1L",       "Bebidas",      1.80,  55),
    (19, "Cerveza pack-6",        "Bebidas",      4.50,  40),
    (20, "Patatas fritas 150g",   "Snacks",       1.30,  60),
    (21, "Aceite oliva 1L",       "Despensa",     4.90,  25),
    (22, "Arroz largo 1kg",       "Despensa",     1.20,  35),
    (23, "Pasta espagueti 500g",  "Despensa",     0.90,  40),
    (24, "Detergente ropa 3L",    "Limpieza",     6.50,  20),
    (25, "Papel higiénico x12",   "Higiene",      3.80,  30),
]

# ── Tiendas ───────────────────────────────────────────────────────────────────
STORES = [
    (1, "SuperFresh Centro",   "Madrid",    "Urbana"),
    (2, "SuperFresh Norte",    "Madrid",    "Suburbana"),
    (3, "SuperFresh Sur",      "Sevilla",   "Urbana"),
    (4, "SuperFresh Levante",  "Valencia",  "Urbana"),
    (5, "SuperFresh Playa",    "Alicante",  "Costera"),
]

# ── Eventos especiales ────────────────────────────────────────────────────────
SPECIAL_EVENTS = {
    # (mes, día): multiplicador
    (1, 1):  1.4,   # Año Nuevo
    (1, 6):  1.3,   # Reyes
    (2, 14): 1.2,   # San Valentín
    (3, 19): 1.2,   # San José
    (4, 18): 1.5,   # Semana Santa (viernes)
    (4, 19): 1.6,   # Semana Santa (sábado)
    (5, 1):  1.3,   # Día del Trabajo
    (6, 24): 1.2,   # San Juan
    (7, 4):  1.1,   # Verano
    (8, 15): 1.3,   # Asunción
    (10, 12):1.2,   # Hispanidad
    (11, 1): 1.2,   # Todos los Santos
    (12, 6): 1.3,   # Constitución
    (12, 24):2.0,   # Nochebuena
    (12, 25):1.8,   # Navidad
    (12, 31):2.2,   # Nochevieja
}

# ── Factores estacionales por mes (bebidas, frutas, etc.) ─────────────────────
CATEGORY_SEASONAL: dict[str, list[float]] = {
    #                    E     F     M     A     M     J     J     A     S     O     N     D
    "Bebidas":        [0.70, 0.72, 0.85, 0.95, 1.10, 1.35, 1.55, 1.50, 1.10, 0.90, 0.75, 0.85],
    "Frutas":         [0.80, 0.82, 0.90, 1.00, 1.10, 1.20, 1.30, 1.25, 1.10, 1.00, 0.85, 0.80],
    "Verduras":       [0.85, 0.85, 0.95, 1.05, 1.10, 1.15, 1.10, 1.05, 1.10, 1.05, 0.90, 0.85],
    "Lácteos":        [1.10, 1.08, 1.05, 1.00, 0.98, 0.95, 0.90, 0.88, 0.95, 1.00, 1.05, 1.15],
    "Panadería":      [1.05, 1.03, 1.00, 1.00, 0.98, 0.97, 0.95, 0.93, 0.98, 1.02, 1.05, 1.10],
    "Carnes":         [0.95, 0.95, 1.00, 1.05, 1.05, 1.10, 1.10, 1.05, 1.00, 0.95, 0.95, 1.10],
    "Snacks":         [0.90, 0.90, 0.95, 1.00, 1.05, 1.15, 1.25, 1.20, 1.05, 1.00, 0.92, 1.10],
    "Despensa":       [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "Limpieza":       [1.05, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.05, 1.05],
    "Higiene":        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}

# ── Tiendas costeras con más demanda de bebidas en verano ─────────────────────
STORE_MODIFIERS: dict[str, dict[str, float]] = {
    "Costera": {"Bebidas": 1.4, "Snacks": 1.3},
    "Suburbana": {"Carnes": 1.1, "Lácteos": 1.1},
}

# ── Promociones ───────────────────────────────────────────────────────────────
# Lista de (product_id, fecha_inicio, fecha_fin, descuento_pct, nombre)
PROMOTIONS_DEF = [
    (16, "2023-07-01", "2023-07-31", 20, "Verano hidratado"),
    (17, "2023-07-01", "2023-07-31", 15, "Refresco de verano"),
    (4,  "2023-09-04", "2023-09-10", 10, "Vuelta al cole"),
    (20, "2023-10-28", "2023-11-04", 25, "Halloween snacks"),
    (1,  "2023-12-20", "2024-01-05", 10, "Navidad lácteos"),
    (13, "2024-04-01", "2024-04-07", 15, "Semana Santa BBQ"),
    (7,  "2024-06-15", "2024-06-30", 20, "Verano frutas"),
    (19, "2024-07-01", "2024-08-31", 15, "Verano cervecero"),
    (11, "2024-09-15", "2024-09-30", 10, "Ensaladas otoño"),
    (24, "2024-11-25", "2024-12-02", 30, "Black Friday limpieza"),
    (21, "2025-01-10", "2025-01-20", 12, "Enero saludable"),
    (17, "2025-07-01", "2025-07-31", 15, "Refresco verano 25"),
    (19, "2025-08-01", "2025-08-31", 10, "Agosto playero"),
    (4,  "2025-09-01", "2025-09-07", 10, "Vuelta al cole 25"),
    (24, "2025-11-28", "2025-12-05", 30, "Black Friday 25"),
]

# ── Factores climáticos por mes (simplificado) ────────────────────────────────
CLIMATE_FACTORS: dict[str, list[float]] = {
    # precipitación alta = más paraguas/hot drinks, menos bebidas frías
    "Bebidas": [0.85, 0.85, 0.90, 0.92, 1.00, 1.15, 1.30, 1.30, 1.10, 0.95, 0.88, 0.80],
}


def _day_of_week_factor(weekday: int) -> float:
    """Viernes y sábado más ventas, lunes menos."""
    factors = [0.85, 0.88, 0.92, 0.95, 1.10, 1.25, 1.05]
    return factors[weekday]


def _build_promotions_lookup(start_date: date, end_date: date) -> dict[tuple, float]:
    lookup: dict[tuple, float] = {}
    for pid, s, e, disc, _ in PROMOTIONS_DEF:
        sd = date.fromisoformat(s)
        ed = date.fromisoformat(e)
        cur = sd
        while cur <= ed:
            if start_date <= cur <= end_date:
                lookup[(cur, pid)] = 1 + disc / 100 * 1.5   # aumento demanda por promo
            cur += timedelta(days=1)
    return lookup


def generate_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genera y retorna DataFrames: products, stores, promotions, weather, sales."""

    start_date = date(2023, 1, 1)
    end_date   = date(2025, 12, 31)
    promo_lookup = _build_promotions_lookup(start_date, end_date)

    # Products
    df_products = pd.DataFrame(PRODUCTS, columns=["product_id", "name", "category", "base_price", "base_demand"])

    # Stores
    df_stores = pd.DataFrame(STORES, columns=["store_id", "name", "city", "store_type"])

    # Promotions
    promo_rows = []
    for pid, s, e, disc, pname in PROMOTIONS_DEF:
        promo_rows.append({"product_id": pid, "start_date": s, "end_date": e,
                           "discount_pct": disc, "promotion_name": pname})
    df_promotions = pd.DataFrame(promo_rows)

    # Weather (temperatura media + lluvia por ciudad y mes)
    weather_rows = []
    for store_id, _, city, _ in STORES:
        for year in [2023, 2024, 2025]:
            for month in range(1, 13):
                base_temp = {"Madrid": [6,7,11,14,18,24,28,28,22,16,10,6],
                             "Sevilla": [11,12,15,18,22,28,32,32,26,20,14,11],
                             "Valencia": [11,12,14,17,20,25,28,28,25,20,15,12],
                             "Alicante": [12,13,15,18,21,26,29,29,26,21,16,13]}
                rain = {"Madrid":    [40,35,25,40,45,20,10,10,30,45,50,50],
                        "Sevilla":   [65,50,35,45,25,10,2,2,20,60,70,65],
                        "Valencia":  [30,25,30,35,30,20,8,10,35,65,50,35],
                        "Alicante":  [20,18,25,28,25,15,5,5,30,55,40,25]}
                temp = base_temp[city][month-1] + random.gauss(0, 1.5)
                rain_mm = rain[city][month-1] * random.uniform(0.7, 1.3)
                weather_rows.append({"store_id": store_id, "year": year, "month": month,
                                     "avg_temp_c": round(temp, 1),
                                     "rain_mm": round(rain_mm, 1)})
    df_weather = pd.DataFrame(weather_rows)

    # Sales
    sales = []
    current = start_date
    prod_map = {p[0]: p for p in PRODUCTS}
    store_map = {s[0]: s for s in STORES}

    while current <= end_date:
        month_idx = current.month - 1
        event_mult = SPECIAL_EVENTS.get((current.month, current.day), 1.0)
        dow_factor = _day_of_week_factor(current.weekday())

        for store_id, _, _, store_type in STORES:
            for pid, pname, category, base_price, base_demand in PRODUCTS:
                seasonal = CATEGORY_SEASONAL.get(category, [1.0]*12)[month_idx]
                store_mod = STORE_MODIFIERS.get(store_type, {}).get(category, 1.0)
                promo_mod = promo_lookup.get((current, pid), 1.0)

                # Descuento si hay promo
                price = base_price
                discount = 0
                if promo_mod > 1.0:
                    for p_pid, ps, pe, disc, _ in PROMOTIONS_DEF:
                        if p_pid == pid:
                            sd = date.fromisoformat(ps)
                            ed = date.fromisoformat(pe)
                            if sd <= current <= ed:
                                discount = disc
                                price = round(base_price * (1 - disc / 100), 2)
                                break

                # Número de unidades con ruido gaussiano
                mean_demand = base_demand * seasonal * event_mult * dow_factor * store_mod * promo_mod * random.uniform(0.9, 1.1)
                units = max(0, int(np.random.normal(mean_demand, mean_demand * 0.12)))
                revenue = round(units * price, 2)

                if units > 0:
                    sales.append({
                        "sale_date": current.isoformat(),
                        "store_id": store_id,
                        "product_id": pid,
                        "units_sold": units,
                        "unit_price": price,
                        "discount_pct": discount,
                        "revenue": revenue,
                    })
        current += timedelta(days=1)

    df_sales = pd.DataFrame(sales)
    return df_products, df_stores, df_promotions, df_weather, df_sales


def save_to_db(df_products, df_stores, df_promotions, df_weather, df_sales):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df_products.to_sql("products", conn, if_exists="replace", index=False)
    df_stores.to_sql("stores", conn, if_exists="replace", index=False)
    df_promotions.to_sql("promotions", conn, if_exists="replace", index=False)
    df_weather.to_sql("weather", conn, if_exists="replace", index=False)
    df_sales.to_sql("sales", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[generate_data] Base de datos guardada en {DB_PATH}")
    print(f"[generate_data] Registros de ventas: {len(df_sales):,}")


def save_csvs(df_products, df_stores, df_promotions, df_weather, df_sales):
    df_products.to_csv(os.path.join(DATA_DIR, "products.csv"), index=False)
    df_stores.to_csv(os.path.join(DATA_DIR, "stores.csv"), index=False)
    df_promotions.to_csv(os.path.join(DATA_DIR, "promotions.csv"), index=False)
    df_weather.to_csv(os.path.join(DATA_DIR, "weather.csv"), index=False)
    df_sales.to_csv(os.path.join(DATA_DIR, "sales.csv"), index=False)
    print("[generate_data] CSVs exportados.")


def main():
    print("[generate_data] Generando datos sintéticos de SuperFresh…")
    dfs = generate_all()
    save_to_db(*dfs)
    save_csvs(*dfs)
    return dfs


if __name__ == "__main__":
    main()
