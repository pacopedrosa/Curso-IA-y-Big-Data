"""
Script principal — arranca la API FastAPI y el dashboard Streamlit.
Uso: python run.py
"""
import os
import sys
import subprocess
import signal
import time

ROOT = os.path.abspath(os.path.dirname(__file__))

API_PORT  = 8002
DASH_PORT = 8502


def check_init():
    required = [
        os.path.join(ROOT, "database", "superfresh.db"),
        os.path.join(ROOT, "database", "rf_model.pkl"),
        os.path.join(ROOT, "database", "gb_model.pkl"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("[run] Faltan archivos. Ejecuta primero: python database/init_db.py")
        for m in missing:
            print(f"  ✗ {m}")
        sys.exit(1)


def main():
    check_init()

    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", str(API_PORT), "--reload"],
        cwd=ROOT,
    )
    print(f"[run] API arrancada en http://localhost:{API_PORT}")
    print(f"[run] Documentación interactiva: http://localhost:{API_PORT}/docs")

    time.sleep(2)  # Espera a que la API levante

    dash_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.join(ROOT, "app", "streamlit_app.py"),
         "--server.port", str(DASH_PORT),
         "--server.headless", "true"],
        cwd=ROOT,
    )
    print(f"[run] Dashboard arrancado en http://localhost:{DASH_PORT}")
    print("\n[run] Pulsa Ctrl+C para detener ambos servicios.\n")

    def shutdown(sig, frame):
        print("\n[run] Deteniendo servicios…")
        api_proc.terminate()
        dash_proc.terminate()
        api_proc.wait()
        dash_proc.wait()
        print("[run] Servicios detenidos.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    api_proc.wait()


if __name__ == "__main__":
    main()
