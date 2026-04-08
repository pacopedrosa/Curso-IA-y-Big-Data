"""
Interfaz visual del sistema de recomendación de cursos.
Construida con Streamlit.

Ejecutar con:
  streamlit run app/streamlit_app.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import sqlite3
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

# ── Configuración ──────────────────────────────────────────────────────────────
API_URL = "http://localhost:8001"
DB_PATH = os.path.join(ROOT, "database", "edtech.db")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

st.set_page_config(
    page_title="EduRecom · Sistema de Recomendación",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 0.5rem;
}
.rec-card {
    background: #f8f9ff;
    border-left: 4px solid #667eea;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.5rem;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers API ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_users(limit: int = 500) -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/users", params={"limit": limit}, timeout=5)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_courses() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/courses", timeout=5)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_recommendations(user_id: str, top_n: int = 5) -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/users/{user_id}/recommendations",
                         params={"top_n": top_n}, timeout=10)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_cluster_info(user_id: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/users/{user_id}/cluster", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_clusters_summary() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/clusters/summary", timeout=5)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=120)
def fetch_metrics(sample_size: int = 50) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/metrics", params={"sample_size": sample_size}, timeout=30)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_status() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.ok
    except Exception:
        return False


# ── Helpers BD (para gráficos que no requieren API) ───────────────────────────

@st.cache_data(ttl=600)
def load_cluster_pca_data() -> pd.DataFrame | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        df_feat = pd.read_sql("""
            SELECT
                i.user_id,
                AVG(i.rating) as avg_rating,
                AVG(i.progress_pct) as avg_progress,
                SUM(i.sessions) as total_sessions,
                AVG(i.session_duration) as avg_session_duration,
                AVG(i.completed) as completed_ratio
            FROM interactions i
            GROUP BY i.user_id
        """, conn)
        df_clusters = pd.read_sql("SELECT user_id, kmeans_cluster, dbscan_cluster FROM user_clusters", conn)
        conn.close()

        if df_feat.empty or df_clusters.empty:
            return None

        merged = df_feat.merge(df_clusters, on="user_id")
        feat_cols = ["avg_rating", "avg_progress", "total_sessions",
                     "avg_session_duration", "completed_ratio"]
        X = merged[feat_cols].fillna(0).values
        pca = PCA(n_components=2, random_state=42)
        X2d = pca.fit_transform(X)
        merged["PC1"] = X2d[:, 0]
        merged["PC2"] = X2d[:, 1]
        merged["kmeans_cluster"] = merged["kmeans_cluster"].astype(str)
        return merged, pca.explained_variance_ratio_
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_interactions_stats() -> pd.DataFrame | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT c.category, AVG(i.rating) as avg_rating,
                   COUNT(*) as n_interactions, AVG(i.progress_pct) as avg_progress
            FROM interactions i
            JOIN courses c ON i.course_id = c.course_id
            GROUP BY c.category
        """, conn)
        conn.close()
        return df
    except Exception:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=70)
    st.title("EduRecom")
    st.caption("Sistema de Recomendación de Cursos")
    st.divider()

    online = api_status()
    if online:
        st.success("API conectada", icon="✅")
    else:
        st.error("API no disponible. Inicia el servidor FastAPI.", icon="🔴")
        st.code("cd api && uvicorn main:app --port 8001", language="bash")

    st.divider()
    page = st.radio(
        "Navegación",
        ["🏠 Inicio", "🔍 Recomendaciones", "📊 Análisis de Clusters", "📈 Métricas del Sistema"],
        label_visibility="collapsed",
    )

# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA: INICIO
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Inicio":
    st.title("🎓 EduRecom · Sistema de Recomendación de Cursos")
    st.markdown(
        "Plataforma de recomendación personalizada que combina **clustering no supervisado** "
        "(K-Means + DBSCAN) con **filtrado colaborativo** para sugerir cursos relevantes a cada usuario."
    )

    col1, col2, col3, col4 = st.columns(4)

    users = fetch_users()
    courses = fetch_courses()
    summary = fetch_clusters_summary()

    with col1:
        st.metric("👥 Usuarios", len(users))
    with col2:
        st.metric("📚 Cursos", len(courses))
    with col3:
        st.metric("🔵 Clusters K-Means", len(summary))
    with col4:
        cats = {c["category"] for c in courses}
        st.metric("🏷️ Categorías", len(cats))

    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Distribución de cursos por categoría")
        if courses:
            df_c = pd.DataFrame(courses)
            cat_count = df_c.groupby("category").size().reset_index(name="count")
            fig_pie = px.pie(cat_count, names="category", values="count",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             hole=0.35)
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.subheader("Interacciones por categoría")
        stats = load_interactions_stats()
        if stats is not None:
            fig_bar = px.bar(
                stats.sort_values("n_interactions", ascending=False),
                x="category", y="n_interactions",
                color="avg_rating",
                color_continuous_scale="Viridis",
                labels={"n_interactions": "Nº interacciones", "avg_rating": "Rating medio"},
            )
            fig_bar.update_layout(margin=dict(t=10, b=10), xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA: RECOMENDACIONES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Recomendaciones":
    st.title("🔍 Recomendaciones Personalizadas")

    users = fetch_users()
    if not users:
        st.warning("No se pudieron cargar los usuarios. Comprueba que la API esté activa.")
        st.stop()

    user_ids = [u["user_id"] for u in users]

    col_sel, col_top = st.columns([3, 1])
    with col_sel:
        selected_user = st.selectbox("Selecciona un usuario", user_ids)
    with col_top:
        top_n = st.slider("Top N recomendaciones", 3, 15, 5)

    if selected_user:
        # Perfil del usuario
        user_data = next((u for u in users if u["user_id"] == selected_user), {})
        cluster_info = fetch_cluster_info(selected_user)

        col_u1, col_u2, col_u3, col_u4 = st.columns(4)
        col_u1.metric("ID", user_data.get("user_id", "—"))
        col_u2.metric("Perfil", user_data.get("profile", "—").replace("_", " ").title())
        col_u3.metric("Seniority", user_data.get("seniority", "—").title())
        col_u4.metric("Cluster K-Means",
                      cluster_info["kmeans_cluster"] if cluster_info else "N/A")

        st.divider()
        st.subheader(f"📌 Top {top_n} recomendaciones para {selected_user}")

        with st.spinner("Generando recomendaciones..."):
            recs = fetch_recommendations(selected_user, top_n)

        if not recs:
            st.info("No se generaron recomendaciones para este usuario.")
        else:
            LEVEL_COLOR = {"básico": "#28a745", "intermedio": "#fd7e14", "avanzado": "#dc3545"}

            for i, rec in enumerate(recs, 1):
                color = LEVEL_COLOR.get(rec["level"], "#6c757d")
                st.markdown(f"""
                <div class="rec-card">
                  <strong>#{i} &nbsp; {rec['name']}</strong>
                  &nbsp;&nbsp;
                  <span class="badge" style="background:{color};color:white">{rec['level']}</span>
                  &nbsp;
                  <span class="badge" style="background:#e9ecef;color:#495057">{rec['category']}</span>
                  <br/>
                  <small>⭐ Rating predicho: <b>{rec['predicted_rating']:.2f}</b>
                  &nbsp;·&nbsp; Rating medio plataforma: {rec['avg_rating']}</small>
                </div>
                """, unsafe_allow_html=True)

            # Gráfico de barras de puntuaciones predichas
            df_recs = pd.DataFrame(recs)
            fig_recs = px.bar(
                df_recs,
                x="name", y="predicted_rating",
                color="category",
                labels={"name": "Curso", "predicted_rating": "Rating predicho"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_recs.update_layout(xaxis_tickangle=-25, showlegend=True,
                                   margin=dict(t=20, b=80))
            st.plotly_chart(fig_recs, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA: ANÁLISIS DE CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Análisis de Clusters":
    st.title("📊 Análisis de Segmentación de Usuarios")

    result = load_cluster_pca_data()

    if result is None:
        st.warning("No hay datos de clustering disponibles. Ejecuta primero `init_db.py`.")
        st.stop()

    df_pca, var_ratio = result

    st.subheader("Visualización PCA de Clusters K-Means")
    st.caption(
        f"Varianza explicada: PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%  "
        f"(total={sum(var_ratio)*100:.1f}%)"
    )

    fig_scatter = px.scatter(
        df_pca, x="PC1", y="PC2", color="kmeans_cluster",
        hover_data=["user_id", "avg_rating", "avg_progress", "completed_ratio"],
        labels={"kmeans_cluster": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.75,
    )
    fig_scatter.update_traces(marker_size=7)
    fig_scatter.update_layout(
        xaxis_title=f"PC1 ({var_ratio[0]*100:.1f}% var.)",
        yaxis_title=f"PC2 ({var_ratio[1]*100:.1f}% var.)",
        legend_title="Cluster K-Means",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.subheader("Resumen de clusters")

    summary = fetch_clusters_summary()
    if summary:
        df_sum = pd.DataFrame(summary)
        df_sum.columns = ["Cluster", "Nº Usuarios", "Rating medio", "Progreso medio (%)", "Categoría top"]

        col_t, col_chart = st.columns([1, 1])
        with col_t:
            st.dataframe(df_sum, use_container_width=True, hide_index=True)
        with col_chart:
            fig_users = px.bar(
                df_sum,
                x="Cluster", y="Nº Usuarios",
                color="Categoría top",
                labels={"Cluster": "Cluster K-Means"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_users.update_layout(xaxis_type="category", margin=dict(t=10))
            st.plotly_chart(fig_users, use_container_width=True)

    # Radar de características por cluster
    if summary:
        st.subheader("Perfil de cada cluster (radar)")
        df_radar = df_pca.groupby("kmeans_cluster")[
            ["avg_rating", "avg_progress", "completed_ratio", "total_sessions", "avg_session_duration"]
        ].mean().reset_index()

        categories = ["avg_rating", "avg_progress", "completed_ratio",
                      "total_sessions", "avg_session_duration"]
        # Normalizar 0-1 para el radar
        for col in categories:
            min_v, max_v = df_radar[col].min(), df_radar[col].max()
            df_radar[col + "_norm"] = (df_radar[col] - min_v) / (max_v - min_v + 1e-9)

        norm_cols = [c + "_norm" for c in categories]
        cat_labels = ["Rating\nmedio", "Progreso\nmedio", "Completados\n(%)",
                      "Sesiones\ntotales", "Duración\nsesión"]

        fig_radar = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, row in df_radar.iterrows():
            vals = row[norm_cols].tolist()
            vals += [vals[0]]  # cerrar el polígono
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=cat_labels + [cat_labels[0]],
                fill="toself",
                name=f"Cluster {int(row['kmeans_cluster'])}",
                line_color=colors[i % len(colors)],
                opacity=0.65,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA: MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Métricas del Sistema":
    st.title("📈 Evaluación del Sistema")

    st.markdown("""
    | Métrica | Descripción |
    |---|---|
    | **Índice de Silhouette** | Mide separación y cohesión de los clusters `[-1, 1]`. Valores > 0.4 son aceptables. |
    | **Precision@K** | Proporción de las K recomendaciones con rating real ≥ 3.5. |
    """)

    sample_size = st.slider("Tamaño de muestra para Precision@K", 20, 200, 50, step=10)

    if st.button("⟳ Calcular métricas", type="primary"):
        with st.spinner("Calculando... (puede tardar ~20 s)"):
            metrics = fetch_metrics(sample_size)

        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Precision@5",
                f"{metrics['precision_at_5']:.2%}",
                help="Porcentaje medio de recomendaciones relevantes entre las 5 primeras"
            )
            col2.metric("Desv. estándar Precision@5", f"±{metrics['precision_std']:.2%}")
            col3.metric("Usuarios evaluados", metrics["n_evaluated"])

            if metrics.get("km_silhouette"):
                sil = metrics["km_silhouette"]
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sil,
                    title={"text": "Índice de Silhouette (K-Means)"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar":  {"color": "#667eea"},
                        "steps": [
                            {"range": [0, 0.25], "color": "#ffcccc"},
                            {"range": [0.25, 0.50], "color": "#ffe0b2"},
                            {"range": [0.50, 0.75], "color": "#c8e6c9"},
                            {"range": [0.75, 1.00], "color": "#b2dfdb"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3}, "value": 0.4},
                    },
                    number={"suffix": "", "valueformat": ".4f"},
                ))
                gauge.update_layout(height=300, margin=dict(t=30, b=10))
                st.plotly_chart(gauge, use_container_width=True)
        else:
            st.error("No se pudieron obtener métricas. Revisa que la API esté en ejecución.")
    else:
        st.info("Pulsa el botón para calcular las métricas en tiempo real.")
