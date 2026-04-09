"""
Script de inicialización de la base de datos y entrenamiento de modelos para SuperFresh.
Ejecutar una sola vez antes de levantar la aplicación.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from data.generate_data import main as generate_data
from models.prediction import run_model_pipeline


def main():
    print("=" * 60)
    print("  SuperFresh — Inicialización del sistema")
    print("=" * 60)

    db_path = os.path.join(ROOT, "database", "superfresh.db")

    if os.path.exists(db_path):
        print(f"\n[init] Base de datos ya existe: {db_path}")
        resp = input("[init] ¿Regenerar datos? (s/N): ").strip().lower()
        if resp != "s":
            print("[init] Se omite la generación de datos.")
        else:
            generate_data()
    else:
        generate_data()

    models_exist = all(
        os.path.exists(os.path.join(ROOT, "database", f))
        for f in ["rf_model.pkl", "gb_model.pkl", "metrics.pkl"]
    )

    if models_exist:
        print("\n[init] Modelos ya entrenados.")
        resp = input("[init] ¿Reentrenar modelos? (s/N): ").strip().lower()
        if resp != "s":
            print("[init] Se omite el entrenamiento.")
        else:
            run_model_pipeline()
    else:
        print("\n[init] Entrenando modelos… (puede tardar 1-2 minutos)")
        metrics_rf, metrics_gb = run_model_pipeline()
        print("\n[init] Métricas finales:")
        print(f"  Random Forest:      {metrics_rf}")
        print(f"  Gradient Boosting:  {metrics_gb}")

    print("\n[init] ✅ Sistema listo. Ejecuta 'python run.py' para arrancar la aplicación.")


if __name__ == "__main__":
    main()
