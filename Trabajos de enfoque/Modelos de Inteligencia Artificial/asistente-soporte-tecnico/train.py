"""
train.py
--------
Script de entrenamiento del clasificador de intenciones.
Ejecutar desde la raíz del proyecto:

    python train.py

Genera el modelo serializado en app/data/model.pkl y muestra
las métricas de evaluación en consola.
"""

import os
import sys
import time

# Asegurar que la raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.intent_classifier import IntentClassifier

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_INTENTS_PATH = os.path.join(_BASE_DIR, "app", "data", "intents.json")
_MODEL_PATH   = os.path.join(_BASE_DIR, "app", "data", "model.pkl")


def format_report(report: dict) -> str:
    """Formatea el classification_report en una tabla legible."""
    lines = [
        f"\n{'Intención':<30} {'Precisión':>10} {'Recall':>10} {'F1':>10} {'Muestras':>10}",
        "-" * 64,
    ]
    for label, metrics in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(metrics, dict):
            lines.append(
                f"{label:<30} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
                f"{metrics['f1-score']:>10.3f} {int(metrics['support']):>10}"
            )
    lines.append("-" * 64)
    avg = report.get("weighted avg", {})
    lines.append(
        f"{'weighted avg':<30} {avg.get('precision', 0):>10.3f} {avg.get('recall', 0):>10.3f} "
        f"{avg.get('f1-score', 0):>10.3f}"
    )
    return "\n".join(lines)


def main():
    print("=" * 64)
    print("  Entrenamiento del Asistente Virtual de Soporte Técnico")
    print("=" * 64)

    if not os.path.exists(_INTENTS_PATH):
        print(f"\n[ERROR] No se encontró el dataset: {_INTENTS_PATH}")
        sys.exit(1)

    classifier = IntentClassifier()

    print(f"\n📂 Dataset:  {_INTENTS_PATH}")
    print("⚙️  Iniciando entrenamiento...\n")

    start = time.perf_counter()
    metrics = classifier.train(_INTENTS_PATH)
    elapsed = time.perf_counter() - start

    print(f"✅ Entrenamiento completado en {elapsed:.2f}s")
    print(f"   Intenciones: {metrics['num_intents']}")
    print(f"   Muestras de entrenamiento: {metrics['num_samples']}")
    print(f"\n📊 Validación cruzada (5-fold):")
    print(f"   Accuracy media: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    print(f"\n📊 Reporte de clasificación (sobre datos de entrenamiento):")
    print(format_report(metrics["report"]))

    # Guardar modelo
    classifier.save(_MODEL_PATH)
    print(f"\n💾 Modelo guardado en: {_MODEL_PATH}")

    # Prueba interactiva básica
    print("\n🧪 Prueba rápida del modelo:")
    test_cases = [
        "no tengo internet",
        "mi pc va muy lento",
        "olvidé mi contraseña",
        "la impresora no funciona",
        "creo que tengo un virus",
    ]
    for text in test_cases:
        intent, confidence = classifier.predict(text)
        print(f"  '{text}' → {intent} ({confidence:.0%})")

    print("\n✨ Modelo listo. Ejecuta la API con:")
    print("   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9050\n")


if __name__ == "__main__":
    main()
