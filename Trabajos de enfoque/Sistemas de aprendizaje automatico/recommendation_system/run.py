#!/usr/bin/env python3
"""
Script de inicio todo-en-uno.
1. Inicializa la BD (genera datos + clustering) si no existe.
2. Arranca la API FastAPI en segundo plano.
3. Lanza la interfaz Streamlit.
"""
import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "database", "edtech.db")


def main():
    # ── Paso 1: inicializar BD ────────────────────────────────────────────────
    if not os.path.exists(DB_PATH):
        print("Base de datos no encontrada. Inicializando...")
        init_script = os.path.join(ROOT, "database", "init_db.py")
        subprocess.run([sys.executable, init_script], check=True)
    else:
        print("Base de datos encontrada. Saltando inicialización.")

    # ── Paso 2: arrancar API ──────────────────────────────────────────────────
    print("\nArrancando API FastAPI en http://localhost:8000 ...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8001"],
        cwd=ROOT,
    )
    time.sleep(4)  # esperar a que la API arranque

    # ── Paso 3: arrancar Streamlit ────────────────────────────────────────────
    print("Lanzando interfaz Streamlit en http://localhost:8501 ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
             "--server.port", "8501", "--server.headless", "true"],
            cwd=ROOT,
        )
    finally:
        api_proc.terminate()
        print("API detenida.")


if __name__ == "__main__":
    main()
