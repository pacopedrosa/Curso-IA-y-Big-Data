"""
Script de inicialización de la base de datos y entrenamiento de modelos para SuperFresh.
Ejecutar una sola vez antes de levantar la aplicación.
"""
import os
import sys
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from data.generate_data import main as generate_data
from models.prediction import run_model_pipeline


def main():
    parser = argparse.ArgumentParser(description="SuperFresh — Inicialización del sistema")
    parser.add_argument("--tune", action="store_true",
                        help="Activar ajuste de hiperparámetros en el entrenamiento (más lento)")
    parser.add_argument("--force", action="store_true",
                        help="Forzar regeneración de datos y reentrenamiento sin preguntar")
    args = parser.parse_args()

    print("=" * 60)
    print("  SuperFresh — Inicialización del sistema")
    print("=" * 60)

    db_path = os.path.join(ROOT, "database", "superfresh.db")

    if os.path.exists(db_path):
        print(f"\n[init] Base de datos ya existe: {db_path}")
        if args.force:
            generate_data()
        else:
            resp = input("[init] ¿Regenerar datos? (s/N): ").strip().lower()
            if resp == "s":
                generate_data()
            else:
                print("[init] Se omite la generación de datos.")
    else:
        generate_data()

    models_exist = all(
        os.path.exists(os.path.join(ROOT, "database", f))
        for f in ["rf_model.pkl", "gb_model.pkl", "metrics.pkl"]
    )

    if models_exist and not args.force:
        print("\n[init] Modelos ya entrenados.")
        resp = input("[init] ¿Reentrenar modelos? (s/N): ").strip().lower()
        if resp != "s":
            print("[init] Se omite el entrenamiento.")
            _print_summary()
            return
    
    tune_msg = " (con ajuste de hiperparámetros)" if args.tune else ""
    print(f"\n[init] Entrenando modelos{tune_msg}… (puede tardar 2-5 minutos)")
    metrics_rf, metrics_gb = run_model_pipeline(tune_hyperparams=args.tune)
    print("\n[init] Métricas finales:")
    print(f"  Random Forest:      {metrics_rf}")
    print(f"  Gradient Boosting:  {metrics_gb}")

    _print_summary()


def _print_summary():
    print("\n[init] ✅ Sistema listo.")
    print("       Ejecuta 'python run.py' para arrancar API + Dashboard.")
    print("       Ejecuta 'python data/spark_processing.py' para el análisis Spark.")
    print("       Ejecuta 'python database/storage.py --help' para exportar a PostgreSQL/MongoDB.")


if __name__ == "__main__":
    main()
