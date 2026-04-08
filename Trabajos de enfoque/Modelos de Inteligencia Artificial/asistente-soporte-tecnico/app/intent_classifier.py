"""
intent_classifier.py
--------------------
Clasificador de intenciones basado en un pipeline de scikit-learn:
  TF-IDF Vectorizer  →  Regresión Logística

El modelo se entrena con el dataset intents.json y puede guardarse en
disco para reutilizarse sin reentrenar en cada arranque.
"""

import json
import os
import pickle
import random
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from app.nlp_processor import NLPProcessor


class IntentClassifier:
    """
    Clasifica la intención del usuario a partir de su mensaje usando
    TF-IDF + Regresión Logística con preprocesamiento PLN en español.
    """

    # Umbral mínimo de confianza para aceptar una predicción
    CONFIDENCE_THRESHOLD = 0.35

    def __init__(self):
        self.processor = NLPProcessor()
        self.pipeline = self._build_pipeline()
        self.intents: dict = {}          # tag → intent dict completo
        self.label_to_intent: dict = {}  # índice → tag
        self.intent_to_label: dict = {}  # tag → índice
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(self, intents_path: str) -> dict:
        """
        Carga el dataset, entrena el pipeline y devuelve métricas.
        Retorna: dict con 'report' (classification_report) y 'cv_score'.
        """
        patterns, labels = self._load_training_data(intents_path)

        label_indices = [self.intent_to_label[l] for l in labels]
        self.pipeline.fit(patterns, label_indices)
        self.is_trained = True

        # Evaluación: reporte de clasificación + validación cruzada
        predictions = self.pipeline.predict(patterns)
        target_names = [self.label_to_intent[i] for i in sorted(self.label_to_intent)]
        report = classification_report(
            label_indices,
            predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        cv_scores = cross_val_score(self.pipeline, patterns, label_indices, cv=5)

        return {
            "report": report,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "num_intents": len(self.intents),
            "num_samples": len(patterns),
        }

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predice la intención del texto dado.
        Devuelve: (tag de intención, confianza 0-1)
        Si la confianza está por debajo de CONFIDENCE_THRESHOLD devuelve 'desconocido'.
        """
        if not self.is_trained:
            raise RuntimeError("El clasificador no ha sido entrenado. Ejecuta train() primero.")

        processed = self.processor.preprocess(text)
        proba = self.pipeline.predict_proba([processed])[0]
        label_idx = int(np.argmax(proba))
        confidence = float(proba[label_idx])

        if confidence < self.CONFIDENCE_THRESHOLD:
            return "desconocido", confidence

        return self.label_to_intent[label_idx], confidence

    def get_response(self, intent: str) -> str:
        """Selecciona aleatoriamente una respuesta del intent dado."""
        if intent in self.intents:
            return random.choice(self.intents[intent]["responses"])
        return (
            "Lo siento, no he comprendido tu consulta. Puedes reformularla o escribir "
            "'hablar con técnico' para contactar con soporte especializado."
        )

    # ------------------------------------------------------------------
    # Persistencia del modelo
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serializa el modelo entrenado en disco."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "intents": self.intents,
            "label_to_intent": self.label_to_intent,
            "intent_to_label": self.intent_to_label,
            "is_trained": self.is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        """Carga el modelo desde disco."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.pipeline = payload["pipeline"]
        self.intents = payload["intents"]
        self.label_to_intent = payload["label_to_intent"]
        self.intent_to_label = payload["intent_to_label"]
        self.is_trained = payload["is_trained"]

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pipeline() -> Pipeline:
        """Construye el pipeline de sklearn: TF-IDF → LogReg."""
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),   # unigramas y bigramas
                        max_features=8000,
                        sublinear_tf=True,    # suavizado logarítmico de TF
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=5.0,                # regularización L2
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def _load_training_data(self, intents_path: str) -> tuple[list[str], list[str]]:
        """Lee el JSON y genera las listas de patrones y etiquetas."""
        with open(intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.intents = {intent["tag"]: intent for intent in data["intents"]}

        unique_tags = sorted(self.intents.keys())
        self.label_to_intent = dict(enumerate(unique_tags))
        self.intent_to_label = {tag: idx for idx, tag in enumerate(unique_tags)}

        patterns: list[str] = []
        labels: list[str] = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                processed = self.processor.preprocess(pattern)
                patterns.append(processed)
                labels.append(intent["tag"])

        return patterns, labels
