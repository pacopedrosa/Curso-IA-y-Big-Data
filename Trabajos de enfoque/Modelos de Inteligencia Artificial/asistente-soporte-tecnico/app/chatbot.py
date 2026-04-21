"""
chatbot.py
----------
Núcleo del Asistente Virtual de Soporte Técnico.
Orquesta el clasificador de intenciones y el motor de diagnóstico,
gestionando el estado de cada sesión de conversación.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from app.diagnostics import DiagnosticEngine
from app.intent_classifier import IntentClassifier

# Rutas relativas al directorio raíz del proyecto
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INTENTS_PATH = os.path.join(_BASE_DIR, "app", "data", "intents.json")
_MODEL_PATH = os.path.join(_BASE_DIR, "app", "data", "model.pkl")

# Intenciones que desencadenan un flujo de diagnóstico guiado
DIAGNOSTIC_INTENTS = {"problema_wifi", "computadora_lenta", "no_enciende"}

# Tiempo máximo de inactividad antes de expirar una sesión (minutos)
SESSION_TTL_MINUTES = 30


@dataclass
class ConversationTurn:
    """Registro de un turno de conversación (usuario + asistente)."""
    timestamp: str
    user_message: str
    bot_response: str
    intent: str
    confidence: float


@dataclass
class Session:
    """Estado de una sesión de usuario."""
    session_id: str
    history: list[ConversationTurn] = field(default_factory=list)
    turn_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)

    def add_turn(self, user_msg: str, bot_resp: str, intent: str, confidence: float) -> None:
        self.history.append(
            ConversationTurn(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                user_message=user_msg,
                bot_response=bot_resp,
                intent=intent,
                confidence=confidence,
            )
        )
        self.turn_count += 1
        self.last_activity = datetime.now()


# Respuesta cuando el clasificador no supera el umbral de confianza
_UNKNOWN_RESPONSE = (
    "Lo siento, no he comprendido tu consulta. Puedes:\n\n"
    "• Reformular la pregunta con más detalle\n"
    "• Escribir 'ayuda' para ver los temas disponibles\n"
    "• Escribir 'técnico' para hablar con soporte especializado"
)

_HELP_RESPONSE = (
    "Puedo ayudarte con los siguientes temas:\n\n"
    "🌐 **Red/WiFi** — problemas de conexión a internet\n"
    "🐌 **PC lento** — rendimiento y lentitud del equipo\n"
    "🖨️ **Impresora** — impresora sin conexión, atascos\n"
    "🔑 **Contraseña** — recuperar o cambiar contraseñas\n"
    "💻 **No enciende** — el equipo no arranca o pantalla negra\n"
    "🦠 **Virus** — eliminación de malware\n"
    "📧 **Correo** — problemas con email corporativo\n"
    "💾 **Backup** — copias de seguridad y recuperación de datos\n"
    "🔄 **Actualizaciones** — Windows Update y drivers\n"
    "📦 **Software** — instalación de programas\n\n"
    "Describe tu problema y te ayudaré."
)


class Chatbot:
    """
    Asistente virtual de soporte técnico.
    Gestiona el ciclo completo: clasificación → diagnóstico (si aplica) → respuesta.
    """

    def __init__(self):
        self.classifier = IntentClassifier()
        self.diagnostic_engine = DiagnosticEngine()
        self._sessions: dict[str, Session] = {}
        self._load_model()

    # ------------------------------------------------------------------
    # API pública principal
    # ------------------------------------------------------------------

    def process(self, session_id: str, user_message: str) -> dict:
        """
        Procesa el mensaje del usuario y devuelve la respuesta del asistente.

        Retorna:
            {
              "response": str,   — texto de respuesta
              "intent": str,     — intención detectada
              "confidence": float — confianza del clasificador
            }
        """
        self._cleanup_expired_sessions()
        session = self._get_or_create_session(session_id)
        user_message = user_message.strip()

        # Comandos especiales
        if self._is_help_request(user_message):
            response = _HELP_RESPONSE
            intent, confidence = "ayuda", 1.0

        # Si hay un diagnóstico en curso, continuar el flujo
        elif self.diagnostic_engine.is_active(session_id):
            response, intent, confidence = self._handle_diagnostic(session_id, user_message)

        # Nueva consulta: clasificar y responder
        else:
            response, intent, confidence = self._classify_and_respond(session_id, user_message)

        session.add_turn(user_message, response, intent, confidence)
        return {"response": response, "intent": intent, "confidence": confidence}

    def get_history(self, session_id: str) -> list[dict]:
        """Devuelve el historial de conversación de una sesión."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [
            {
                "timestamp": t.timestamp,
                "user": t.user_message,
                "bot": t.bot_response,
                "intent": t.intent,
                "confidence": round(t.confidence, 3),
            }
            for t in session.history
        ]

    def clear_session(self, session_id: str) -> None:
        """Elimina la sesión y su diagnóstico activo si lo hubiera."""
        self._sessions.pop(session_id, None)
        self.diagnostic_engine.cancel(session_id)

    # ------------------------------------------------------------------
    # Métodos privados de procesamiento
    # ------------------------------------------------------------------

    def _classify_and_respond(
        self, session_id: str, message: str
    ) -> tuple[str, str, float]:
        """Clasifica la intención y genera una respuesta."""
        intent, confidence = self.classifier.predict(message)

        if intent == "desconocido":
            return _UNKNOWN_RESPONSE, "desconocido", confidence

        # Intenciones con flujo de diagnóstico guiado
        if intent in DIAGNOSTIC_INTENTS:
            first_question = self.diagnostic_engine.start(session_id, intent)
            if first_question:
                return first_question, intent, confidence

        response = self.classifier.get_response(intent)
        return response, intent, confidence

    def _handle_diagnostic(
        self, session_id: str, user_message: str
    ) -> tuple[str, str, float]:
        """Continúa un flujo de diagnóstico activo."""
        # Permitir cancelar el diagnóstico
        if any(kw in user_message.lower() for kw in ("cancelar", "salir", "volver", "parar")):
            self.diagnostic_engine.cancel(session_id)
            return "Diagnóstico cancelado. ¿En qué más puedo ayudarte?", "cancelar", 1.0

        response, finished = self.diagnostic_engine.process(session_id, user_message)
        intent = "diagnostico_fin" if finished else "diagnostico"
        return response, intent, 1.0

    # ------------------------------------------------------------------
    # Helpers de sesión y modelo
    # ------------------------------------------------------------------

    def _get_or_create_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def _cleanup_expired_sessions(self) -> None:
        """Elimina sesiones inactivas que superan SESSION_TTL_MINUTES."""
        cutoff = datetime.now() - timedelta(minutes=SESSION_TTL_MINUTES)
        expired = [
            sid for sid, session in self._sessions.items()
            if session.last_activity < cutoff
        ]
        for sid in expired:
            self.clear_session(sid)

    def _load_model(self) -> None:
        """Carga el modelo preentrenado o entrena uno nuevo si no existe."""
        if os.path.exists(_MODEL_PATH):
            try:
                self.classifier.load(_MODEL_PATH)
                return
            except Exception:
                pass  # Si falla la carga, re-entrenamos

        # Entrenar y guardar
        if os.path.exists(_INTENTS_PATH):
            self.classifier.train(_INTENTS_PATH)
            self.classifier.save(_MODEL_PATH)

    @staticmethod
    def _is_help_request(message: str) -> bool:
        """Detecta peticiones de ayuda/menú."""
        keywords = {"ayuda", "help", "menu", "menú", "opciones", "temas", "qué puedes hacer"}
        return message.lower().strip() in keywords
