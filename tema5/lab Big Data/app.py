"""
app.py — Servidor Flask para la Fase 2 (AWS EC2)
Expone endpoints HTTP para que Make pueda disparar los scripts remotamente.

Endpoints:
  GET  /status          → comprueba que el servidor está activo
  GET  /extraer         → ejecuta extraer.py
  GET  /descargar-csv   → devuelve el archivo precios_libros.csv
  GET  /predecir        → ejecuta predecir.py (requiere credenciales Google)
  GET  /subir-google    → ejecuta data2google.py
"""

import os
import subprocess
import sys
from flask import Flask, jsonify, send_file, abort

app = Flask(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_CSV = os.path.join(BASE_DIR, "data", "precios_libros.csv")
PYTHON      = sys.executable   # ruta al intérprete de Python actual


def ejecutar_script(nombre_script: str) -> dict:
    """Ejecuta un script Python en un subproceso y devuelve stdout/stderr."""
    ruta = os.path.join(BASE_DIR, nombre_script)
    if not os.path.exists(ruta):
        return {"ok": False, "error": f"Script no encontrado: {nombre_script}"}

    resultado = subprocess.run(
        [PYTHON, ruta],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
        timeout=300,   # máximo 5 minutos
    )
    return {
        "ok":     resultado.returncode == 0,
        "stdout": resultado.stdout[-4000:],   # últimas 4000 chars para no saturar
        "stderr": resultado.stderr[-2000:],
        "codigo": resultado.returncode,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    """Health-check básico."""
    return jsonify({"status": "ok", "message": "Servidor activo"}), 200


@app.route("/extraer")
def extraer():
    """Ejecuta el script de scraping."""
    print("Ejecutando extraer.py …")
    resultado = ejecutar_script("extraer.py")
    codigo_http = 200 if resultado["ok"] else 500
    return jsonify(resultado), codigo_http


@app.route("/subir-google")
def subir_google():
    """Ejecuta data2google.py para subir el CSV a Google Sheets."""
    print("Ejecutando data2google.py …")
    resultado = ejecutar_script("data2google.py")
    codigo_http = 200 if resultado["ok"] else 500
    return jsonify(resultado), codigo_http


@app.route("/predecir")
def predecir():
    """Ejecuta el script de predicción ML."""
    print("Ejecutando predecir.py …")
    resultado = ejecutar_script("predecir.py")
    codigo_http = 200 if resultado["ok"] else 500
    return jsonify(resultado), codigo_http


@app.route("/descargar-csv")
def descargar_csv():
    """Devuelve el archivo CSV como descarga."""
    if not os.path.exists(ARCHIVO_CSV):
        abort(404, description="CSV no encontrado. Ejecuta primero /extraer.")
    return send_file(
        ARCHIVO_CSV,
        mimetype="text/csv",
        as_attachment=True,
        download_name="precios_libros.csv",
    )


# ── Arranque ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # En producción (AWS) se recomienda usar Gunicorn en lugar de este servidor
    # Ejemplo: gunicorn -b 0.0.0.0:80 app:app
    app.run(host="0.0.0.0", port=80, debug=False)
