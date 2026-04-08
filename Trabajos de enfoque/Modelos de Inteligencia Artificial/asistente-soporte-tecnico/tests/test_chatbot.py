"""
test_chatbot.py
---------------
Suite de tests para el Asistente Virtual de Soporte Técnico.
Cubre: NLPProcessor, IntentClassifier, DiagnosticEngine y Chatbot.

Ejecutar con pytest desde la raíz del proyecto:
    pytest tests/ -v
"""

import os
import sys

import pytest

# Asegurar que la raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chatbot import Chatbot
from app.diagnostics import DiagnosticEngine
from app.intent_classifier import IntentClassifier
from app.nlp_processor import NLPProcessor

_BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INTENTS_PATH = os.path.join(_BASE_DIR, "app", "data", "intents.json")


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def nlp():
    """Instancia del procesador NLP (compartida en todo el módulo)."""
    return NLPProcessor()


@pytest.fixture(scope="module")
def trained_classifier():
    """Clasificador entrenado con los intents del proyecto."""
    clf = IntentClassifier()
    clf.train(_INTENTS_PATH)
    return clf


@pytest.fixture(scope="module")
def chatbot():
    """Chatbot completo con el modelo cargado."""
    return Chatbot()


# ===========================================================================
# Tests de NLPProcessor
# ===========================================================================

class TestNLPProcessor:

    def test_preprocess_lowercase(self, nlp):
        result = nlp.preprocess("HOLA TENGO UN PROBLEMA")
        assert result == result.lower()

    def test_preprocess_removes_punctuation(self, nlp):
        result = nlp.preprocess("¡Hola! ¿Cómo estás?")
        assert "!" not in result and "?" not in result

    def test_preprocess_removes_stopwords(self, nlp):
        result = nlp.preprocess("no tengo internet en mi casa")
        # "en" y "mi" son stopwords → no deben aparecer
        tokens = result.split()
        assert "en" not in tokens
        assert "mi" not in tokens

    def test_preprocess_returns_string(self, nlp):
        result = nlp.preprocess("prueba de procesamiento")
        assert isinstance(result, str)

    def test_preprocess_empty_string(self, nlp):
        result = nlp.preprocess("")
        assert result == ""

    def test_extract_keywords(self, nlp):
        kws = nlp.extract_keywords("el ordenador va muy lento")
        assert isinstance(kws, list)
        assert len(kws) > 0

    def test_normalize_accents(self, nlp):
        # El preprocesado debe tratar "internet" e "ínternet" igual
        r1 = nlp.preprocess("conexion a internet")
        r2 = nlp.preprocess("conexión a ínternet")
        assert r1 == r2


# ===========================================================================
# Tests de IntentClassifier
# ===========================================================================

class TestIntentClassifier:

    def test_is_trained_after_training(self, trained_classifier):
        assert trained_classifier.is_trained is True

    def test_predict_returns_tuple(self, trained_classifier):
        intent, confidence = trained_classifier.predict("no tengo internet")
        assert isinstance(intent, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.parametrize("message, expected_intent", [
        ("no tengo internet",               "problema_wifi"),
        ("mi pc va muy lento",              "computadora_lenta"),
        ("olvidé mi contraseña",            "reinicio_contrasena"),
        ("la impresora no funciona",        "problema_impresora"),
        ("creo que tengo un virus",         "virus_malware"),
        ("hola necesito ayuda",             "saludo"),
        ("gracias por tu ayuda",            "agradecimiento"),
        ("quiero hablar con un técnico",    "soporte_humano"),
        ("el ordenador no enciende",        "no_enciende"),
    ])
    def test_classify_common_queries(self, trained_classifier, message, expected_intent):
        intent, confidence = trained_classifier.predict(message)
        assert intent == expected_intent, (
            f"Para '{message}': esperado '{expected_intent}', obtenido '{intent}' ({confidence:.2%})"
        )

    def test_low_confidence_returns_desconocido(self, trained_classifier):
        # Texto completamente irrelevante
        intent, _ = trained_classifier.predict("xkcd mango fibonacci")
        assert intent == "desconocido"

    def test_get_response_known_intent(self, trained_classifier):
        response = trained_classifier.get_response("problema_wifi")
        assert isinstance(response, str)
        assert len(response) > 10

    def test_get_response_unknown_intent(self, trained_classifier):
        response = trained_classifier.get_response("intent_inexistente")
        assert "soporte" in response.lower() or "reformular" in response.lower()

    def test_intents_loaded(self, trained_classifier):
        assert len(trained_classifier.intents) > 5

    def test_save_and_load(self, trained_classifier, tmp_path):
        model_path = str(tmp_path / "test_model.pkl")
        trained_classifier.save(model_path)
        assert os.path.exists(model_path)

        new_clf = IntentClassifier()
        new_clf.load(model_path)
        assert new_clf.is_trained

        intent, _ = new_clf.predict("no tengo internet")
        assert intent == "problema_wifi"


# ===========================================================================
# Tests de DiagnosticEngine
# ===========================================================================

class TestDiagnosticEngine:

    def test_start_returns_question_for_known_intent(self):
        engine = DiagnosticEngine()
        result = engine.start("session_1", "problema_wifi")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 10

    def test_start_returns_none_for_unknown_intent(self):
        engine = DiagnosticEngine()
        result = engine.start("session_1", "saludo")
        assert result is None

    def test_is_active_after_start(self):
        engine = DiagnosticEngine()
        engine.start("session_2", "problema_wifi")
        assert engine.is_active("session_2") is True

    def test_process_affirmative_response(self):
        engine = DiagnosticEngine()
        engine.start("session_3", "problema_wifi")
        response, finished = engine.process("session_3", "sí")
        assert isinstance(response, str)
        assert len(response) > 5

    def test_process_negative_response(self):
        engine = DiagnosticEngine()
        engine.start("session_4", "problema_wifi")
        response, finished = engine.process("session_4", "no")
        assert isinstance(response, str)
        assert finished is True  # respuesta final

    def test_cancel_removes_session(self):
        engine = DiagnosticEngine()
        engine.start("session_5", "computadora_lenta")
        engine.cancel("session_5")
        assert engine.is_active("session_5") is False

    def test_process_inactive_session(self):
        engine = DiagnosticEngine()
        response, finished = engine.process("nonexistent_session", "sí")
        assert finished is True


# ===========================================================================
# Tests de Chatbot (integración)
# ===========================================================================

class TestChatbot:

    def test_process_returns_required_keys(self, chatbot):
        result = chatbot.process("test_session_1", "hola")
        assert "response" in result
        assert "intent" in result
        assert "confidence" in result

    def test_process_greeting(self, chatbot):
        result = chatbot.process("test_session_2", "hola")
        assert result["intent"] == "saludo"
        assert len(result["response"]) > 10

    def test_process_wifi_problem_starts_diagnostic(self, chatbot):
        result = chatbot.process("test_session_wifi", "no tengo internet")
        assert result["intent"] == "problema_wifi"
        # Debe iniciar un diagnóstico guiado con una pregunta
        assert "?" in result["response"]
        chatbot.clear_session("test_session_wifi")

    def test_process_help_command(self, chatbot):
        result = chatbot.process("test_session_3", "ayuda")
        assert result["intent"] == "ayuda"
        assert "red" in result["response"].lower() or "wifi" in result["response"].lower()

    def test_diagnostic_flow_complete(self, chatbot):
        sid = "test_diag_flow"
        # Paso 1: iniciar diagnóstico
        chatbot.process(sid, "mi pc va muy lento")
        # Paso 2: responder al flujo
        result = chatbot.process(sid, "inicio")
        assert isinstance(result["response"], str)
        # Paso 3: responder suivante
        result = chatbot.process(sid, "no")
        assert isinstance(result["response"], str)
        chatbot.clear_session(sid)

    def test_cancel_diagnostic(self, chatbot):
        sid = "test_cancel"
        chatbot.process(sid, "no tengo internet")
        result = chatbot.process(sid, "cancelar")
        assert "cancelado" in result["response"].lower()
        chatbot.clear_session(sid)

    def test_get_history(self, chatbot):
        sid = "test_history"
        chatbot.process(sid, "hola")
        chatbot.process(sid, "no tengo internet")
        history = chatbot.get_history(sid)
        assert len(history) == 2
        assert "user" in history[0]
        assert "bot" in history[0]
        chatbot.clear_session(sid)

    def test_clear_session(self, chatbot):
        sid = "test_clear"
        chatbot.process(sid, "hola")
        chatbot.clear_session(sid)
        history = chatbot.get_history(sid)
        assert history == []

    def test_unknown_query_response(self, chatbot):
        result = chatbot.process("test_unknown", "xkcd fibonacci quantum mango")
        assert result["intent"] == "desconocido"
        assert len(result["response"]) > 10
