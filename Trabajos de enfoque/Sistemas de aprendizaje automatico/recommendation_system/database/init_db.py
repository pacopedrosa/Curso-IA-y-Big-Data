"""
Script de inicialización de la base de datos.
Ejecuta en orden: generación de datos → clustering → (listo para CF).
"""
import os
import sys

BASE = os.path.dirname(__file__)
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, ROOT)

from data.generate_data import main as generate
from models.clustering import run_clustering_pipeline

DB_PATH = os.path.join(BASE, "edtech.db")
PLOTS_DIR = os.path.join(ROOT, "app", "static")


def init_db():
    print("=" * 50)
    print("PASO 1: Generando datos sintéticos...")
    print("=" * 50)
    generate()

    print()
    print("=" * 50)
    print("PASO 2: Entrenando modelos de clustering...")
    print("=" * 50)
    result = run_clustering_pipeline(db_path=DB_PATH, n_clusters=5, plots_dir=PLOTS_DIR)

    print()
    print("=" * 50)
    print("Inicialización completada.")
    print(f"  K-Means Silhouette : {result['km_silhouette']:.4f}")
    print(f"  DBSCAN  Silhouette : {result['db_silhouette']:.4f}")
    print(f"  BD guardada en    : {DB_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    init_db()
