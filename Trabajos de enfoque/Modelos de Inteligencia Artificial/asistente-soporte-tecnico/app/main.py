"""
main.py
-------
API REST del Asistente Virtual de Soporte Técnico.
Construida con FastAPI y sirve también la interfaz web estática.

Endpoints:
  GET  /              → Interfaz web (chat UI)
  POST /api/chat      → Enviar mensaje al asistente
  GET  /api/health    → Estado del servicio
  GET  /api/faqs      → Listado de preguntas frecuentes
  GET  /api/history/{session_id} → Historial de conversación
  DELETE /api/session/{session_id} → Limpiar sesión
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from app.chatbot import Chatbot

# ---------------------------------------------------------------------------
# Rutas de archivos estáticos
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATIC_DIR = os.path.join(_BASE_DIR, "static")

# ---------------------------------------------------------------------------
# Estado global de la aplicación
# ---------------------------------------------------------------------------
_chatbot: Chatbot | None = None
_startup_time: float = time.time()

# Métricas de rendimiento (rolling, en memoria)
_metrics: dict = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}

# Almacén de valoraciones de usuarios (en memoria, privacidad por diseño)
_feedback_store: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el chatbot al arrancar y libera recursos al cerrar."""
    global _chatbot, _startup_time
    _startup_time = time.time()
    _chatbot = Chatbot()
    yield
    _chatbot = None


# ---------------------------------------------------------------------------
# Middleware: mide el tiempo de respuesta de cada petición
# ---------------------------------------------------------------------------

class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """Registra la latencia de cada request y añade la cabecera X-Response-Time-Ms."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Actualizar métricas globales (solo para rutas /api/)
        if request.url.path.startswith("/api/"):
            _metrics["count"] += 1
            _metrics["total_ms"] += elapsed_ms
            if elapsed_ms > _metrics["max_ms"]:
                _metrics["max_ms"] = elapsed_ms

        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Asistente Virtual de Soporte Técnico",
    description=(
        "API REST para el asistente virtual de soporte técnico basado en PLN. "
        "Utiliza TF-IDF + Regresión Logística para clasificar intenciones y "
        "un motor de diagnóstico guiado para problemas técnicos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware de tiempo de respuesta
app.add_middleware(ResponseTimeMiddleware)

# CORS: configurable mediante la variable de entorno ALLOWED_ORIGINS.
# En producción: ALLOWED_ORIGINS=https://midominio.com
# En desarrollo (por defecto): se permiten todos los orígenes.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_ALLOWED_ORIGINS = ["*"] if _raw_origins == "*" else [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# Servir archivos estáticos (CSS, JS, imágenes)
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Modelos de datos (Pydantic)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, description="ID único de sesión del usuario")
    message: str = Field(..., min_length=1, max_length=1000, description="Mensaje del usuario")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    confidence: float


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, description="ID de sesión del usuario")
    message_id: str = Field(..., description="Identificador del mensaje valorado")
    rating: int = Field(..., ge=-1, le=1, description="1 = útil, -1 = no útil")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float
    avg_response_ms: float
    total_requests: int
    satisfaction_rate: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    """Sirve la interfaz web del chatbot."""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Interfaz web no encontrada.")
    return FileResponse(index_path)


@app.post("/api/chat", response_model=ChatResponse, summary="Enviar mensaje al asistente")
async def chat(request: ChatRequest):
    """
    Procesa el mensaje del usuario y devuelve la respuesta del asistente.
    El campo `session_id` identifica la sesión; usa siempre el mismo valor
    para mantener el contexto de la conversación.
    """
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="El servicio no está disponible.")

    result = _chatbot.process(request.session_id, request.message)
    return ChatResponse(
        session_id=request.session_id,
        response=result["response"],
        intent=result["intent"],
        confidence=round(result["confidence"], 4),
    )


@app.get("/api/health", response_model=HealthResponse, summary="Estado del servicio")
async def health():
    """Comprueba el estado del servicio e incluye métricas de rendimiento y satisfacción."""
    model_loaded = _chatbot is not None and _chatbot.classifier.is_trained

    total = _metrics["count"]
    avg_ms = _metrics["total_ms"] / total if total > 0 else 0.0

    total_fb = len(_feedback_store)
    positive_fb = sum(1 for f in _feedback_store if f["rating"] > 0)
    satisfaction = round(positive_fb / total_fb, 4) if total_fb > 0 else 0.0

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
        uptime_seconds=round(time.time() - _startup_time, 1),
        avg_response_ms=round(avg_ms, 2),
        total_requests=total,
        satisfaction_rate=satisfaction,
    )


@app.get("/api/faqs", summary="Listado de preguntas frecuentes")
async def get_faqs():
    """
    Devuelve las categorías de consultas disponibles con ejemplos de preguntas.
    """
    if _chatbot is None or not _chatbot.classifier.is_trained:
        raise HTTPException(status_code=503, detail="El servicio no está disponible.")

    # Intenciones de sistema que no se muestran al usuario
    hidden = {"saludo", "despedida", "agradecimiento"}

    faqs = []
    for tag, intent_data in _chatbot.classifier.intents.items():
        if tag in hidden:
            continue
        faqs.append(
            {
                "category": tag.replace("_", " ").title(),
                "tag": tag,
                "description": intent_data.get("description", ""),
                "example_questions": intent_data["patterns"][:3],
            }
        )

    return {"total": len(faqs), "faqs": faqs}


@app.get("/api/history/{session_id}", summary="Historial de conversación")
async def get_history(session_id: str):
    """Devuelve el historial de mensajes de una sesión."""
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="El servicio no está disponible.")
    history = _chatbot.get_history(session_id)
    return {"session_id": session_id, "turns": len(history), "history": history}


@app.delete("/api/session/{session_id}", summary="Limpiar sesión")
async def clear_session(session_id: str):
    """Elimina la sesión y su contexto de diagnóstico asociado."""
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="El servicio no está disponible.")
    _chatbot.clear_session(session_id)
    return {"session_id": session_id, "cleared": True}


@app.post("/api/feedback", summary="Valorar respuesta del asistente")
async def submit_feedback(request: FeedbackRequest):
    """
    Registra la valoración del usuario sobre una respuesta del asistente.
    Permite medir la satisfacción y detectar respuestas que necesitan mejora.

    - **rating = 1**  → respuesta útil (👍)
    - **rating = -1** → respuesta no útil (👎)
    """
    _feedback_store.append({
        "session_id": request.session_id,
        "message_id": request.message_id,
        "rating": request.rating,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    total = len(_feedback_store)
    positive = sum(1 for f in _feedback_store if f["rating"] > 0)
    satisfaction = round(positive / total, 4) if total > 0 else 0.0

    return {
        "received": True,
        "total_feedback": total,
        "satisfaction_rate": satisfaction,
    }
