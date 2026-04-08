"""
Módulo de clustering de usuarios.
Implementa K-Means y DBSCAN para segmentar usuarios según sus patrones de uso.

Métricas de evaluación:
  - Índice de Silhouette: mide cohesión y separación de los clusters [-1, 1].
  - Inercia (K-Means): suma de distancias cuadradas al centroide (método del codo).
"""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pickle
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "edtech.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "database")


# ── Utilidades ─────────────────────────────────────────────────────────────────

def load_data(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    users = pd.read_sql("SELECT * FROM users", conn)
    courses = pd.read_sql("SELECT * FROM courses", conn)
    interactions = pd.read_sql("SELECT * FROM interactions", conn)
    conn.close()
    return users, courses, interactions


def build_user_feature_matrix(interactions: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz de características por usuario a partir de interacciones.
    Características:
      - avg_rating: valoración media dada por el usuario
      - avg_progress: progreso medio en los cursos
      - total_sessions: total de sesiones
      - avg_session_duration: duración media de sesión
      - completed_ratio: ratio de cursos completados
      - Por cada categoría: número de cursos interactuados (normalizado)
    """
    merged = interactions.merge(courses[["course_id", "category"]], on="course_id")

    # Agregados generales
    agg = merged.groupby("user_id").agg(
        avg_rating=("rating", "mean"),
        avg_progress=("progress_pct", "mean"),
        total_sessions=("sessions", "sum"),
        avg_session_duration=("session_duration", "mean"),
        completed_ratio=("completed", "mean"),
        n_courses=("course_id", "nunique"),
    ).reset_index()

    # Preferencias por categoría (counts pivotados)
    cat_pivot = merged.groupby(["user_id", "category"]).size().unstack(fill_value=0).reset_index()
    # Normalizar por fila para obtener proporciones
    cat_cols = [c for c in cat_pivot.columns if c != "user_id"]
    row_sums = cat_pivot[cat_cols].sum(axis=1).replace(0, 1)
    cat_pivot[cat_cols] = cat_pivot[cat_cols].div(row_sums, axis=0)

    features = agg.merge(cat_pivot, on="user_id", how="left").fillna(0)
    return features


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"user_id"}
    return [c for c in df.columns if c not in exclude]


# ── K-Means ────────────────────────────────────────────────────────────────────

def elbow_method(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> tuple[list, list]:
    """Calcula inercia y silhouette para distintos valores de k."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 5) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_scaled)
    return km


# ── DBSCAN ────────────────────────────────────────────────────────────────────

def fit_dbscan(X_scaled: np.ndarray, eps: float = 1.8, min_samples: int = 5,
               pca_components: int = 8) -> np.ndarray:
    """
    Aplica DBSCAN sobre las primeras `pca_components` componentes principales
    para evitar la maldición de la dimensionalidad.
    Etiqueta -1 para ruido/outliers. Devuelve array de etiquetas.
    """
    # Reducción de dimensionalidad antes de DBSCAN
    n_comp = min(pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_pca)
    return labels


# ── Visualizaciones ───────────────────────────────────────────────────────────

def plot_elbow(k_range, inertias: list, silhouettes: list, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2)
    axes[0].set_title("Método del Codo (K-Means)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inercia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "rs-", linewidth=2)
    axes[1].set_title("Índice de Silhouette vs k", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
    return fig


def plot_clusters_pca(X_scaled: np.ndarray, labels: np.ndarray, title: str = "Clusters (PCA)",
                       save_path: str = None):
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({"PC1": X2d[:, 0], "PC2": X2d[:, 1], "Cluster": labels.astype(str)})
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("Set2", n_colors=len(df_plot["Cluster"].unique()))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Cluster", palette=palette,
                    alpha=0.7, s=60, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
    return fig


def plot_cluster_profiles(features: pd.DataFrame, labels: np.ndarray,
                           feature_cols: list[str], save_path: str = None):
    """Muestra el perfil medio de cada cluster (heatmap)."""
    df = features[feature_cols].copy()
    df["Cluster"] = labels
    profile = df.groupby("Cluster")[feature_cols[:8]].mean()  # top 8 features

    fig, ax = plt.subplots(figsize=(12, max(4, len(profile) * 0.8)))
    sns.heatmap(profile, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Perfil medio por Cluster (K-Means)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Pipeline principal ─────────────────────────────────────────────────────────

def run_clustering_pipeline(db_path: str = DB_PATH, n_clusters: int = 5,
                             plots_dir: str = None) -> dict:
    """
    Ejecuta el pipeline completo de clustering.
    Devuelve un dict con modelos, métricas y DataFrame de usuarios con cluster asignado.
    """
    users, courses, interactions = load_data(db_path)
    features = build_user_feature_matrix(interactions, courses)
    feat_cols = get_feature_columns(features)

    X = features[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Selección de k óptimo ──────────────────────────────────────────────────
    k_range = range(2, 11)
    inertias, silhouettes = elbow_method(X_scaled, k_range)
    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"  Mejor k por Silhouette: {best_k}  (score={max(silhouettes):.4f})")
    if n_clusters != best_k:
        print(f"  Usando k={n_clusters} según configuración (best_k={best_k})")

    # ── K-Means ────────────────────────────────────────────────────────────────
    km = fit_kmeans(X_scaled, n_clusters)
    km_labels = km.labels_
    km_sil = silhouette_score(X_scaled, km_labels)
    print(f"  K-Means  | k={n_clusters} | Silhouette={km_sil:.4f}")

    # ── DBSCAN ─────────────────────────────────────────────────────────────────
    db_labels = fit_dbscan(X_scaled, eps=1.8, min_samples=5, pca_components=8)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = int((db_labels == -1).sum())
    db_valid = db_labels != -1
    db_sil = silhouette_score(X_scaled[db_valid], db_labels[db_valid]) if db_valid.sum() > 1 else 0.0
    print(f"  DBSCAN   | clusters={n_clusters_db} | ruido={n_noise} | Silhouette={db_sil:.4f}")

    # ── Guardar modelos ────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    pickle.dump(scaler,  open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))
    pickle.dump(km,      open(os.path.join(MODEL_DIR, "kmeans.pkl"), "wb"))
    pickle.dump(feat_cols, open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb"))

    # ── Asignar clusters a usuarios ────────────────────────────────────────────
    features = features.copy()
    features["kmeans_cluster"] = km_labels
    features["dbscan_cluster"] = db_labels

    # Persistir asignaciones en la BD
    conn = sqlite3.connect(db_path)
    cluster_df = features[["user_id", "kmeans_cluster", "dbscan_cluster"]]
    cluster_df.to_sql("user_clusters", conn, if_exists="replace", index=False)
    conn.close()

    # ── Gráficos ───────────────────────────────────────────────────────────────
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plot_elbow(k_range, inertias, silhouettes,
                   save_path=os.path.join(plots_dir, "elbow_silhouette.png"))
        plot_clusters_pca(X_scaled, km_labels, "Clusters K-Means (PCA 2D)",
                          save_path=os.path.join(plots_dir, "kmeans_pca.png"))
        plot_clusters_pca(X_scaled, db_labels, "Clusters DBSCAN (PCA 2D)",
                          save_path=os.path.join(plots_dir, "dbscan_pca.png"))
        plot_cluster_profiles(features, km_labels, feat_cols,
                              save_path=os.path.join(plots_dir, "cluster_profiles.png"))
        plt.close("all")

    return {
        "features":       features,
        "kmeans_model":   km,
        "scaler":         scaler,
        "feature_cols":   feat_cols,
        "km_silhouette":  km_sil,
        "db_silhouette":  db_sil,
        "best_k":         best_k,
        "inertias":       inertias,
        "silhouettes":    silhouettes,
    }


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    plots = os.path.join(base, "..", "app", "static")
    result = run_clustering_pipeline(plots_dir=plots)
    print(f"\nResumen:")
    print(f"  K-Means Silhouette : {result['km_silhouette']:.4f}")
    print(f"  DBSCAN  Silhouette : {result['db_silhouette']:.4f}")
