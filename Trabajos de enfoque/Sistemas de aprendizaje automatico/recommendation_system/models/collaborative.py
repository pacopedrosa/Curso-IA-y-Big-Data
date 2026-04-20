"""
Filtrado colaborativo basado en similitud de usuarios (User-Based CF).

Algoritmo:
  1. Construir la matriz usuario-curso de ratings.
  2. Calcular la similitud coseno entre todos los pares de usuarios.
  3. Para un usuario objetivo, identificar los k vecinos más cercanos dentro
     del mismo cluster (o en todo el dataset si se prefiere).
  4. Recomendar los cursos mejor valorados por esos vecinos que el usuario
     objetivo aún no ha visto.

Métrica de evaluación: Precision@K
  Porcentaje de recomendaciones relevantes (rating real >= umbral) entre
  las K recomendaciones generadas.
"""
import sqlite3
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "edtech.db")


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_ratings_matrix(db_path: str = DB_PATH) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """
    Devuelve la matriz usuario x curso (ratings), junto con los índices.
    Los ratings faltantes se rellenan con 0.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT user_id, course_id, rating FROM interactions", conn)
    conn.close()

    matrix = df.pivot_table(index="user_id", columns="course_id", values="rating", fill_value=0)
    return matrix, matrix.index, matrix.columns


def load_clusters(db_path: str = DB_PATH) -> pd.DataFrame:
    """Carga la asignación de clusters generada por el módulo de clustering."""
    conn = sqlite3.connect(db_path)
    try:
        clusters = pd.read_sql("SELECT * FROM user_clusters", conn)
    except Exception:
        clusters = pd.DataFrame(columns=["user_id", "kmeans_cluster", "dbscan_cluster"])
    conn.close()
    return clusters


def load_courses(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    courses = pd.read_sql("SELECT * FROM courses", conn)
    conn.close()
    return courses


# ── Similitud ─────────────────────────────────────────────────────────────────

def compute_user_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """Calcula la similitud coseno entre todos los usuarios."""
    sparse = csr_matrix(matrix.values)
    sim_matrix = cosine_similarity(sparse)
    return pd.DataFrame(sim_matrix, index=matrix.index, columns=matrix.index)


# ── Recomendación ─────────────────────────────────────────────────────────────

def get_top_k_neighbors(user_id: str, sim_df: pd.DataFrame, k: int = 10,
                        same_cluster: list[str] = None) -> list[str]:
    """
    Devuelve los k usuarios más similares al usuario objetivo.
    Si same_cluster no es None, filtra solo entre esos usuarios.
    """
    if user_id not in sim_df.index:
        return []
    similarities = sim_df.loc[user_id].drop(index=user_id)
    if same_cluster is not None:
        pool = [u for u in same_cluster if u != user_id and u in similarities.index]
        similarities = similarities.loc[pool]
    return similarities.nlargest(k).index.tolist()


def predict_ratings(user_id: str, neighbors: list[str],
                    matrix: pd.DataFrame, sim_df: pd.DataFrame) -> pd.Series:
    """
    Predice el rating del usuario objetivo para los cursos no vistos,
    usando la media ponderada de ratings de los vecinos.
    """
    if not neighbors:
        return pd.Series(dtype=float)

    seen = set(matrix.loc[user_id][matrix.loc[user_id] > 0].index)
    unseen_courses = [c for c in matrix.columns if c not in seen]

    neighbor_matrix = matrix.loc[neighbors, unseen_courses]
    sims = sim_df.loc[user_id, neighbors].values

    # Evitar división por cero
    non_zero_mask = neighbor_matrix > 0
    weighted_sum = neighbor_matrix.T.dot(sims)
    sim_sum = non_zero_mask.T.dot(np.abs(sims))
    sim_sum = np.where(sim_sum == 0, 1e-9, sim_sum)

    predicted = pd.Series(weighted_sum / sim_sum, index=unseen_courses)
    return predicted.sort_values(ascending=False)


def recommend(user_id: str, matrix: pd.DataFrame, sim_df: pd.DataFrame,
              courses: pd.DataFrame, clusters: pd.DataFrame = None,
              n_neighbors: int = 20, top_n: int = 5) -> pd.DataFrame:
    """
    Genera las top_n recomendaciones de cursos para un usuario.
    Devuelve un DataFrame con course_id, name, category, level, predicted_rating.
    """
    same_cluster = None
    if clusters is not None and not clusters.empty and user_id in clusters["user_id"].values:
        user_cluster = clusters.loc[clusters["user_id"] == user_id, "kmeans_cluster"].values[0]
        same_cluster = clusters.loc[
            clusters["kmeans_cluster"] == user_cluster, "user_id"
        ].tolist()

    neighbors = get_top_k_neighbors(user_id, sim_df, k=n_neighbors, same_cluster=same_cluster)
    if not neighbors:
        neighbors = get_top_k_neighbors(user_id, sim_df, k=n_neighbors)  # fallback sin filtro

    predictions = predict_ratings(user_id, neighbors, matrix, sim_df)
    if predictions.empty:
        return pd.DataFrame()

    top_courses = predictions.head(top_n).reset_index()
    top_courses.columns = ["course_id", "predicted_rating"]
    result = top_courses.merge(courses[["course_id", "name", "category", "level", "avg_rating"]],
                               on="course_id", how="left")
    result["predicted_rating"] = result["predicted_rating"].round(2)
    return result


# ── Métricas ──────────────────────────────────────────────────────────────────

def precision_at_k(recommendations: pd.DataFrame, actual_ratings: pd.Series,
                   k: int = 5, threshold: float = 3.5) -> float:
    """
    Precision@K: fracción de las K recomendaciones con rating real >= threshold.
    actual_ratings: Series con course_id como índice y rating como valor.
    """
    top_k = recommendations.head(k)["course_id"].tolist()
    if not top_k:
        return 0.0
    relevant = sum(1 for cid in top_k
                   if cid in actual_ratings.index and actual_ratings[cid] >= threshold)
    return relevant / k


def evaluate_system(matrix: pd.DataFrame, sim_df: pd.DataFrame,
                    courses: pd.DataFrame, clusters: pd.DataFrame,
                    sample_size: int = 50, k: int = 5,
                    threshold: float = 3.5) -> dict:
    """
    Evalúa el sistema sobre una muestra de usuarios usando holdout multi-ítem.
    Estrategia: para cada usuario se retiran el 25% de sus ítems relevantes
    (rating >= threshold) como conjunto de test. Se mide cuántos de los top-K
    recomendados pertenecen a ese conjunto relevante → Precision@K real.
    Esto es más representativo que leave-one-out con un único ítem.
    """
    eligible = [u for u in matrix.index if (matrix.loc[u] >= threshold).sum() >= 4]
    actual_sample = min(sample_size, len(eligible))
    users_sample = np.random.choice(eligible, size=actual_sample, replace=False)
    precisions = []

    for uid in users_sample:
        seen = matrix.loc[uid]
        relevant_items = seen[seen >= threshold]

        # Holdout del 25% de ítems relevantes (mínimo 1, máximo 5)
        n_test = max(1, min(5, int(len(relevant_items) * 0.25)))
        test_items = relevant_items.sample(n=n_test, random_state=None)

        # Retirar los ítems de test de la matriz temporal
        matrix_temp = matrix.copy()
        matrix_temp.loc[uid, test_items.index] = 0

        recs = recommend(uid, matrix_temp, sim_df, courses, clusters,
                         n_neighbors=20, top_n=k)
        if recs.empty:
            continue

        # Precision@K: fracción de top-K recomendados que están en test_items
        p_k = precision_at_k(recs, test_items, k=k, threshold=threshold)
        precisions.append(p_k)

    if not precisions:
        return {"precision_at_k": 0.0, "std": 0.0, "n_evaluated": 0}

    return {
        "precision_at_k": float(np.mean(precisions)),
        "std":            float(np.std(precisions)),
        "n_evaluated":    len(precisions),
    }


# ── API pública ───────────────────────────────────────────────────────────────

class CollaborativeFilteringModel:
    """
    Wrapper para cargar/usar el modelo de CF sin reentrenar en cada petición.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.matrix: pd.DataFrame = None
        self.sim_df: pd.DataFrame = None
        self.courses: pd.DataFrame = None
        self.clusters: pd.DataFrame = None

    def fit(self):
        self.matrix, _, _ = load_ratings_matrix(self.db_path)
        self.sim_df = compute_user_similarity(self.matrix)
        self.courses = load_courses(self.db_path)
        self.clusters = load_clusters(self.db_path)
        return self

    def recommend(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        return recommend(user_id, self.matrix, self.sim_df,
                         self.courses, self.clusters, top_n=top_n)

    def evaluate(self, sample_size: int = 50, k: int = 5) -> dict:
        return evaluate_system(self.matrix, self.sim_df, self.courses,
                               self.clusters, sample_size=sample_size, k=k)


if __name__ == "__main__":
    model = CollaborativeFilteringModel()
    model.fit()

    # Prueba rápida
    sample_user = model.matrix.index[0]
    recs = model.recommend(sample_user, top_n=5)
    print(f"\nRecomendaciones para {sample_user}:")
    print(recs[["course_id", "name", "category", "predicted_rating"]].to_string(index=False))

    metrics = model.evaluate(sample_size=50)
    print(f"\nPrecision@5 media : {metrics['precision_at_k']:.4f}  ±{metrics['std']:.4f}")
    print(f"Usuarios evaluados: {metrics['n_evaluated']}")
