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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.chatbot import Chatbot

# ---------------------------------------------------------------------------
# Rutas de archivos estáticos
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATIC_DIR = os.path.join(_BASE_DIR, "static")

# ---------------------------------------------------------------------------
# Singleton del chatbot (se inicializa al arrancar la app)
# ---------------------------------------------------------------------------
_chatbot: Chatbot | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el chatbot al arrancar y libera recursos al cerrar."""
    global _chatbot
    _chatbot = Chatbot()
    yield
    _chatbot = None


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

# CORS: permite peticiones desde la UI web (en producción, especificar dominio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


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
    """Comprueba que el servicio y el modelo están operativos."""
    model_loaded = _chatbot is not None and _chatbot.classifier.is_trained
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
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
