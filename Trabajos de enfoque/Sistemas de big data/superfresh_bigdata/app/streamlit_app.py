"""
Cuadro de mandos Streamlit — SuperFresh Sales Intelligence
Visualización en tiempo real de ventas, predicciones y métricas del modelo.
"""
import os
import sys
import requests
import sqlite3
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, timedelta

warnings.filterwarnings("ignore")

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "database", "superfresh.db")
API_URL = "http://localhost:8002"

st.set_page_config(
    page_title="SuperFresh — Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a237e, #283593);
        border-radius: 12px;
        padding: 18px 22px;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers de datos ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_db_data():
    conn = sqlite3.connect(DB_PATH)
    sales    = pd.read_sql("SELECT * FROM sales",    conn, parse_dates=["sale_date"])
    products = pd.read_sql("SELECT * FROM products", conn)
    stores   = pd.read_sql("SELECT * FROM stores",   conn)
    weather  = pd.read_sql("SELECT * FROM weather",  conn)
    conn.close()
    return sales, products, stores, weather


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error conectando con la API: {e}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error en la API: {e}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/shopping-cart--v1.png", width=80)
st.sidebar.title("SuperFresh")
st.sidebar.markdown("**Sales Intelligence Dashboard**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "📈 Predicciones", "📊 Análisis de Ventas", "🤖 Modelos", "⚙️ Configuración"],
)

sales, products, stores, weather = load_db_data()

# ── PÁGINA: INICIO ─────────────────────────────────────────────────────────────
if page == "🏠 Inicio":
    st.title("🛒 SuperFresh — Panel de Control")
    st.markdown("Sistema de predicción de demanda basado en Big Data y Machine Learning")

    # KPIs globales
    total_revenue = sales["revenue"].sum()
    total_units   = sales["units_sold"].sum()
    total_products = sales["product_id"].nunique()
    total_stores   = sales["store_id"].nunique()
    date_range = f"{sales['sale_date'].min().strftime('%d/%m/%Y')} – {sales['sale_date'].max().strftime('%d/%m/%Y')}"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Ingresos totales",   f"€{total_revenue:,.0f}")
    c2.metric("📦 Unidades vendidas",  f"{total_units:,}")
    c3.metric("🏪 Tiendas",            f"{total_stores}")
    c4.metric("🛍️ Productos",          f"{total_products}")
    c5.metric("📅 Período",            date_range)

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Tendencia mensual de ingresos
        monthly = (sales.groupby(sales["sale_date"].dt.to_period("M"))["revenue"]
                   .sum().reset_index())
        monthly["sale_date"] = monthly["sale_date"].dt.to_timestamp()
        fig_trend = px.area(monthly, x="sale_date", y="revenue",
                            title="Evolución mensual de ingresos",
                            labels={"sale_date": "Mes", "revenue": "Ingresos (€)"},
                            color_discrete_sequence=["#3498db"])
        fig_trend.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        # Ventas por categoría
        cat_sales = (sales.merge(products[["product_id", "category"]], on="product_id")
                     .groupby("category")["revenue"].sum().reset_index())
        fig_pie = px.pie(cat_sales, names="category", values="revenue",
                         title="Ingresos por categoría",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Heatmap ventas por día de semana y mes
    st.markdown("#### Mapa de calor — Unidades vendidas por día y mes")
    sales_hw = sales.copy()
    sales_hw["dow"]   = sales_hw["sale_date"].dt.day_name()
    sales_hw["month"] = sales_hw["sale_date"].dt.month_name()
    pivot = sales_hw.pivot_table(values="units_sold", index="dow", columns="month",
                                  aggfunc="mean")
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in dow_order if d in pivot.index])
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    pivot = pivot[[m for m in month_order if m in pivot.columns]]
    fig_heat = px.imshow(pivot, color_continuous_scale="Blues",
                          labels=dict(x="Mes", y="Día", color="Und. medias"),
                          aspect="auto")
    fig_heat.update_layout(height=320)
    st.plotly_chart(fig_heat, use_container_width=True)


# ── PÁGINA: PREDICCIONES ───────────────────────────────────────────────────────
elif page == "📈 Predicciones":
    st.title("📈 Predicciones de Demanda")

    tab1, tab2 = st.tabs(["Predicción puntual", "Previsión de los próximos 30 días"])

    with tab1:
        st.markdown("#### Predice las unidades a vender en una fecha concreta")
        c1, c2, c3 = st.columns(3)
        product_options = {f"{r['name']} (ID {r['product_id']})": r["product_id"]
                          for _, r in products.iterrows()}
        store_options   = {f"{r['name']}": r["store_id"] for _, r in stores.iterrows()}

        sel_product = c1.selectbox("Producto", list(product_options.keys()))
        sel_store   = c2.selectbox("Tienda",   list(store_options.keys()))
        sel_date    = c3.date_input("Fecha a predecir", value=date.today() + timedelta(days=7))
        disc        = st.slider("Descuento (%)", 0, 50, 0)
        sel_model   = st.radio("Modelo", ["random_forest", "gradient_boosting"], horizontal=True)

        if st.button("🔮 Predecir", type="primary"):
            payload = {
                "product_id": product_options[sel_product],
                "store_id":   store_options[sel_store],
                "target_date": sel_date.isoformat(),
                "discount_pct": disc,
                "model": sel_model,
            }
            result = api_post("/predict", payload)
            if result:
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Unidades predichas", f"{result['predicted_units']:.0f}")
                cc2.metric("Ingresos predichos", f"€{result['predicted_revenue']:.2f}")
                cc3.metric("Intervalo inferior", f"{result['confidence_lower']:.0f}")
                cc4.metric("Intervalo superior", f"{result['confidence_upper']:.0f}")
                st.success(f"Modelo usado: **{result['model_used']}**")

    with tab2:
        st.markdown("#### Previsión diaria para los próximos 30 días")
        c1, c2 = st.columns(2)
        sel_product2 = c1.selectbox("Producto", list(product_options.keys()), key="bp")
        sel_store2   = c2.selectbox("Tienda",   list(store_options.keys()),   key="bs")
        disc2 = st.slider("Descuento (%)", 0, 50, 0, key="bd")
        model2 = st.radio("Modelo", ["random_forest", "gradient_boosting"], horizontal=True, key="bm")

        if st.button("📅 Generar previsión de 30 días", type="primary"):
            payload2 = {
                "product_id":  product_options[sel_product2],
                "store_id":    store_options[sel_store2],
                "days_ahead":  30,
                "discount_pct": disc2,
                "model": model2,
            }
            batch = api_post("/predict/batch", payload2)
            if batch and batch.get("forecast"):
                df_fc = pd.DataFrame(batch["forecast"])
                df_fc["date"] = pd.to_datetime(df_fc["date"])

                fig_fc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       subplot_titles=("Unidades previstas", "Ingresos previstos (€)"))
                fig_fc.add_trace(go.Scatter(x=df_fc["date"], y=df_fc["predicted_units"],
                                            mode="lines+markers", name="Unidades",
                                            line=dict(color="#3498db")), row=1, col=1)
                fig_fc.add_trace(go.Bar(x=df_fc["date"], y=df_fc["predicted_revenue"],
                                        name="Ingresos", marker_color="#2ecc71"), row=2, col=1)
                fig_fc.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_fc, use_container_width=True)

                total_fc_units = df_fc["predicted_units"].sum()
                total_fc_rev   = df_fc["predicted_revenue"].sum()
                st.info(f"Total previsto 30 días → **{total_fc_units:.0f} unidades** | **€{total_fc_rev:.2f}**")
                st.dataframe(df_fc.rename(columns={
                    "date": "Fecha",
                    "predicted_units": "Unidades",
                    "predicted_revenue": "Ingresos (€)"
                }), use_container_width=True)


# ── PÁGINA: ANÁLISIS DE VENTAS ─────────────────────────────────────────────────
elif page == "📊 Análisis de Ventas":
    st.title("📊 Análisis de Ventas Históricas")

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)
    cats  = ["Todas"] + sorted(products["category"].unique().tolist())
    sel_cat   = col_f1.selectbox("Categoría", cats)
    store_opts = ["Todas"] + stores["name"].tolist()
    sel_store  = col_f2.selectbox("Tienda", store_opts)
    date_range = col_f3.date_input(
        "Rango de fechas",
        value=(date(2025, 1, 1), date(2025, 12, 31)),
    )

    df_filt = sales.copy()
    if sel_cat != "Todas":
        pids = products[products["category"] == sel_cat]["product_id"].tolist()
        df_filt = df_filt[df_filt["product_id"].isin(pids)]
    if sel_store != "Todas":
        sid = stores[stores["name"] == sel_store]["store_id"].values[0]
        df_filt = df_filt[df_filt["store_id"] == sid]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        df_filt = df_filt[(df_filt["sale_date"] >= pd.Timestamp(date_range[0])) &
                          (df_filt["sale_date"] <= pd.Timestamp(date_range[1]))]

    df_filt = df_filt.merge(products[["product_id", "name", "category"]], on="product_id")

    # KPIs filtrados
    c1, c2, c3 = st.columns(3)
    c1.metric("Ingresos período", f"€{df_filt['revenue'].sum():,.0f}")
    c2.metric("Unidades vendidas", f"{df_filt['units_sold'].sum():,}")
    c3.metric("Ticket medio diario", f"€{df_filt.groupby('sale_date')['revenue'].sum().mean():,.0f}")

    col1, col2 = st.columns(2)

    with col1:
        # Top 10 productos
        top = (df_filt.groupby(["product_id", "name"])["revenue"]
               .sum().reset_index().sort_values("revenue", ascending=True).tail(10))
        fig_top = px.bar(top, x="revenue", y="name", orientation="h",
                          title="Top 10 productos por ingresos",
                          labels={"revenue": "Ingresos (€)", "name": "Producto"},
                          color="revenue", color_continuous_scale="Blues")
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        # Ventas por tienda
        by_store = (df_filt.merge(stores[["store_id", "name"]], on="store_id", suffixes=("","_store"))
                    .groupby("name_store")["revenue"].sum().reset_index())
        fig_stores = px.bar(by_store, x="name_store", y="revenue",
                             title="Ingresos por tienda",
                             labels={"revenue": "Ingresos (€)", "name_store": "Tienda"},
                             color="revenue", color_continuous_scale="Greens")
        st.plotly_chart(fig_stores, use_container_width=True)

    # Tendencia semanal
    weekly = df_filt.resample("W", on="sale_date")["units_sold"].sum().reset_index()
    fig_weekly = px.line(weekly, x="sale_date", y="units_sold",
                          title="Tendencia semanal de unidades vendidas",
                          labels={"sale_date": "Semana", "units_sold": "Unidades"})
    st.plotly_chart(fig_weekly, use_container_width=True)

    # Impacto de descuentos
    st.markdown("#### Impacto de descuentos en ventas")
    disc_df = df_filt.groupby("discount_pct").agg(
        avg_units=("units_sold", "mean")
    ).reset_index()
    fig_disc = px.scatter(disc_df, x="discount_pct", y="avg_units",
                           size="avg_units", color="avg_units",
                           labels={"discount_pct": "Descuento (%)", "avg_units": "Unidades medias"},
                           title="Relación descuento → unidades vendidas",
                           color_continuous_scale="Reds")
    st.plotly_chart(fig_disc, use_container_width=True)


# ── PÁGINA: MODELOS ────────────────────────────────────────────────────────────
elif page == "🤖 Modelos":
    st.title("🤖 Evaluación de Modelos Predictivos")
    st.markdown("Comparativa de métricas entre los modelos entrenados.")

    metrics_data = api_get("/metrics")
    if metrics_data:
        col1, col2 = st.columns(2)

        def show_model_card(title: str, m: dict, col):
            with col:
                st.markdown(f"### {title}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE",  f"{m.get('MAE',  '—')}")
                c2.metric("RMSE", f"{m.get('RMSE', '—')}")
                c3.metric("R²",   f"{m.get('R2',   '—')}")
                c4.metric("MAPE", f"{m.get('MAPE', '—')}%")

        show_model_card("🌲 Random Forest",       metrics_data["random_forest"],       col1)
        show_model_card("⚡ Gradient Boosting",    metrics_data["gradient_boosting"],    col2)

        # Radar chart comparación
        m_rf = metrics_data["random_forest"]
        m_gb = metrics_data["gradient_boosting"]

        # R² normalizado para radar
        radar_cats = ["MAE_inv", "RMSE_inv", "R²", "MAPE_inv"]
        max_mae  = max(m_rf["MAE"],  m_gb["MAE"])  or 1
        max_rmse = max(m_rf["RMSE"], m_gb["RMSE"]) or 1
        max_mape = max(m_rf["MAPE"], m_gb["MAPE"]) or 1

        vals_rf = [1 - m_rf["MAE"]/max_mae,  1 - m_rf["RMSE"]/max_rmse,
                   max(0, m_rf["R2"]),        1 - m_rf["MAPE"]/max_mape]
        vals_gb = [1 - m_gb["MAE"]/max_mae,  1 - m_gb["RMSE"]/max_rmse,
                   max(0, m_gb["R2"]),        1 - m_gb["MAPE"]/max_mape]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=vals_rf + [vals_rf[0]], theta=radar_cats + [radar_cats[0]],
                                              fill="toself", name="Random Forest", line_color="#3498db"))
        fig_radar.add_trace(go.Scatterpolar(r=vals_gb + [vals_gb[0]], theta=radar_cats + [radar_cats[0]],
                                              fill="toself", name="Gradient Boosting", line_color="#e74c3c"))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                 title="Comparación radar (mayor = mejor)",
                                 showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Importancia de características
    st.markdown("#### Importancia de características — Random Forest")
    feat_img = os.path.join(ROOT, "app", "static", "feature_importance.png")
    if os.path.exists(feat_img):
        st.image(feat_img, use_column_width=True)
    else:
        st.info("Ejecuta el pipeline de modelos para generar este gráfico.")

    # ARIMA
    st.markdown("#### Predicción ARIMA — Productos representativos")
    product_options = {f"{r['name']}": r["product_id"] for _, r in products.iterrows()}
    sel_p = st.selectbox("Selecciona producto", list(product_options.keys()))
    pid   = product_options[sel_p]
    arima_img = os.path.join(ROOT, "app", "static", f"arima_product_{pid}.png")
    if os.path.exists(arima_img):
        st.image(arima_img, use_column_width=True)
    else:
        st.info(f"No hay gráfico ARIMA generado para este producto (ID {pid}).")

    # Comparación de modelos
    cmp_img = os.path.join(ROOT, "app", "static", "model_comparison.png")
    if os.path.exists(cmp_img):
        st.markdown("#### Comparación de métricas")
        st.image(cmp_img, use_column_width=True)


# ── PÁGINA: CONFIGURACIÓN ──────────────────────────────────────────────────────
elif page == "⚙️ Configuración":
    st.title("⚙️ Estado del Sistema")

    status = api_get("/")
    if status:
        st.success(f"✅ API operativa — {status.get('service','')}")
    else:
        st.error("❌ API no disponible. Asegúrate de que está corriendo en el puerto 8001.")

    st.markdown("### Resumen de datos cargados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros de ventas",  f"{len(sales):,}")
    c2.metric("Productos",            f"{len(products)}")
    c3.metric("Tiendas",              f"{len(stores)}")
    c4.metric("Registros de clima",   f"{len(weather)}")

    st.markdown("### Productos en catálogo")
    st.dataframe(products, use_container_width=True)

    st.markdown("### Tiendas")
    st.dataframe(stores, use_container_width=True)
