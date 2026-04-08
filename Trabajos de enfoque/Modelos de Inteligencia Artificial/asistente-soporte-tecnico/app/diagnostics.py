"""
diagnostics.py
--------------
Motor de diagnóstico técnico guiado por pasos.
Implementa flujos conversacionales (step-by-step) para los problemas
más comunes, permitiendo al asistente hacer preguntas de seguimiento
y ofrecer soluciones personalizadas según las respuestas del usuario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiagnosticStep:
    """Representa un paso dentro de un flujo de diagnóstico."""
    question: str                           # Pregunta que se muestra al usuario
    options: dict[str, "DiagnosticStep | str"]  # Respuesta esperada → siguiente paso o solución final
    fallback: str = ""                      # Mensaje si no se reconoce la respuesta


# ---------------------------------------------------------------------------
# Definición de los flujos de diagnóstico
# ---------------------------------------------------------------------------

DIAGNOSTIC_FLOWS: dict[str, DiagnosticStep] = {

    "problema_wifi": DiagnosticStep(
        question=(
            "Voy a ayudarte a diagnosticar el problema de red paso a paso.\n\n"
            "¿Las luces del router/módem están encendidas? (sí/no)"
        ),
        options={
            "si": DiagnosticStep(
                question="¿Otros dispositivos de tu red (móvil, otro PC) tienen internet? (sí/no)",
                options={
                    "si": (
                        "El problema es específico de tu equipo. Te recomiendo:\n\n"
                        "1️⃣ Panel de control → Redes → Adaptadores → Deshabilita y vuelve a habilitar\n"
                        "2️⃣ Actualiza los drivers de la tarjeta de red\n"
                        "3️⃣ Ejecuta en CMD como administrador:\n"
                        "   `netsh int ip reset` y luego `ipconfig /flushdns`\n"
                        "4️⃣ Reinicia el equipo\n\n"
                        "Si continúa el problema, abre un ticket de incidencia con el equipo de TI."
                    ),
                    "no": (
                        "El problema está en el router o en la línea de internet.\n\n"
                        "1️⃣ Desenchufa el router durante 60 segundos y vuelve a conectarlo\n"
                        "2️⃣ Comprueba que los cables de red estén bien insertados\n"
                        "3️⃣ Llama a tu proveedor de internet para reportar la incidencia\n\n"
                        "Si el router tiene un botón de 'reset', no lo pulses (borraría la configuración)."
                    ),
                },
                fallback="Por favor, responde 'sí' o 'no': ¿otros dispositivos tienen internet?",
            ),
            "no": (
                "El router o módem no responde correctamente. Pasos a seguir:\n\n"
                "1️⃣ Desenchúfalo de la corriente durante 30-60 segundos\n"
                "2️⃣ Verifica que el cable de alimentación esté bien conectado\n"
                "3️⃣ Vuelve a enchufarlo y espera 2 minutos\n"
                "4️⃣ Si las luces siguen apagadas o parpadeando en rojo, contacta a tu ISP\n\n"
                "📞 También puedes llamar al soporte: 900 XXX XXX"
            ),
        },
        fallback="Por favor, responde 'sí' o 'no': ¿las luces del router están encendidas?",
    ),

    "computadora_lenta": DiagnosticStep(
        question=(
            "Vamos a identificar la causa de la lentitud.\n\n"
            "¿El equipo es lento desde el inicio del sistema o solo al usar ciertos programas?\n"
            "Responde: 'inicio' o 'programas'"
        ),
        options={
            "inicio": DiagnosticStep(
                question=(
                    "¿El disco duro aparece al 100% en el Administrador de tareas?\n"
                    "(Ctrl+Alt+Supr → Rendimiento → Disco) — Responde: 'sí' o 'no'"
                ),
                options={
                    "si": (
                        "El cuello de botella es el disco duro. Soluciones:\n\n"
                        "1️⃣ Deshabilita la búsqueda de Windows: services.msc → Windows Search → Desactivar\n"
                        "2️⃣ Deshabilita la indexación en la unidad C: → Propiedades → desmarcar indexación\n"
                        "3️⃣ Verifica que no tengas malware con un análisis completo\n"
                        "4️⃣ Si el disco tiene más de 5 años, considera reemplazarlo por un SSD\n\n"
                        "Un SSD puede multiplicar por 5-10x el rendimiento respecto a un disco tradicional."
                    ),
                    "no": (
                        "El problema de inicio se debe probablemente a programas que arrancan con Windows.\n\n"
                        "1️⃣ Abre el Administrador de tareas → pestaña Inicio\n"
                        "2️⃣ Deshabilita los programas con 'Alto impacto' que no necesites\n"
                        "3️⃣ Asegúrate de tener al menos 10-15% de espacio libre en C:\n"
                        "4️⃣ Ejecuta `sfc /scannow` en CMD como administrador para reparar archivos del sistema"
                    ),
                },
                fallback="Responde 'sí' si el disco está al 100%, o 'no' si no.",
            ),
            "programas": (
                "Si la lentitud ocurre en programas concretos:\n\n"
                "1️⃣ Abre el Administrador de tareas y observa qué proceso consume más CPU/RAM\n"
                "2️⃣ Cierra las aplicaciones que no estés usando\n"
                "3️⃣ Verifica los requisitos mínimos del programa problemático\n"
                "4️⃣ Actualiza el software a la última versión\n"
                "5️⃣ Si la RAM está al 90%+ de uso, cerrar otras aplicaciones o ampliar la RAM\n\n"
                "¿Qué programa en concreto te da problemas? Puedo orientarte mejor."
            ),
        },
        fallback="Por favor responde 'inicio' si es lento al arrancar, o 'programas' si solo con ciertas apps.",
    ),

    "no_enciende": DiagnosticStep(
        question=(
            "Vamos a diagnosticar el problema de arranque.\n\n"
            "¿Cuando pulsas el botón de encendido, los ventiladores/LEDs hacen algo? (sí/no)"
        ),
        options={
            "si": DiagnosticStep(
                question="¿Ves algo en la pantalla (aunque sea brevemente)? (sí/no)",
                options={
                    "si": (
                        "El hardware arranca pero Windows no carga. Soluciones:\n\n"
                        "1️⃣ Inicia en **Modo Seguro**: presiona F8 o Shift durante el arranque\n"
                        "2️⃣ Selecciona 'Reparar el equipo' → 'Reparar inicio'\n"
                        "3️⃣ En el símbolo del sistema de recuperación ejecuta:\n"
                        "   `bootrec /fixmbr` y luego `bootrec /fixboot`\n"
                        "4️⃣ Si hay un mensaje de error específico, anótalo y abre un ticket con ese código."
                    ),
                    "no": (
                        "El equipo arranca pero no da imagen. Posibles causas:\n\n"
                        "1️⃣ Verifica el cable del monitor (HDMI/DisplayPort/VGA)\n"
                        "2️⃣ Prueba con un monitor diferente\n"
                        "3️⃣ Si tiene tarjeta gráfica dedicada, prueba conectar al puerto de la placa base\n"
                        "4️⃣ Comprueba que el monitor esté encendido y en la entrada correcta\n\n"
                        "Si con otro monitor tampoco hay imagen, solicita asistencia técnica."
                    ),
                },
                fallback="Responde 'sí' si ves algo en pantalla, o 'no' si está completamente en negro.",
            ),
            "no": (
                "El equipo no responde en absoluto. Pasos de diagnóstico físico:\n\n"
                "1️⃣ Verifica que el cable de alimentación esté bien conectado a la corriente Y al equipo\n"
                "2️⃣ Prueba en otro enchufe o regleta\n"
                "3️⃣ En portátiles: retira la batería, conecta solo el cargador e intenta encender\n"
                "4️⃣ En sobremesa: verifica el interruptor físico de la fuente de alimentación (parte trasera)\n"
                "5️⃣ Si nada funciona, es probable un fallo de hardware → solicita servicio técnico urgente."
            ),
        },
        fallback="Por favor responde 'sí' o 'no': ¿el equipo hace alguna reacción al encenderse?",
    ),
}


# ---------------------------------------------------------------------------
# Motor de diagnóstico
# ---------------------------------------------------------------------------

class DiagnosticEngine:
    """
    Gestiona el estado de las sesiones de diagnóstico activas.
    Cada sesión se identifica por un session_id y puede estar
    en cualquier paso del flujo correspondiente.
    """

    def __init__(self):
        # session_id → DiagnosticStep actual
        self._active_sessions: dict[str, DiagnosticStep] = {}

    def start(self, session_id: str, intent: str) -> Optional[str]:
        """
        Inicia un flujo de diagnóstico si existe para el intent dado.
        Devuelve la primera pregunta o None si no hay flujo para ese intent.
        """
        if intent not in DIAGNOSTIC_FLOWS:
            return None
        self._active_sessions[session_id] = DIAGNOSTIC_FLOWS[intent]
        return DIAGNOSTIC_FLOWS[intent].question

    def process(self, session_id: str, user_input: str) -> tuple[str, bool]:
        """
        Procesa la respuesta del usuario dentro de un flujo activo.
        Devuelve: (mensaje de respuesta, sesión_finalizada)
        """
        step = self._active_sessions.get(session_id)
        if step is None:
            return "", True  # No hay diagnóstico activo

        # Normalizar entrada del usuario
        key = self._normalize_answer(user_input)

        # Buscar la clave más cercana en las opciones del paso actual
        matched_key = self._match_option(key, step.options)

        if matched_key is None:
            return step.fallback or "No he entendido tu respuesta. Por favor intenta de nuevo.", False

        next_step = step.options[matched_key]

        if isinstance(next_step, str):
            # Respuesta final → finalizar diagnóstico
            del self._active_sessions[session_id]
            return next_step, True
        else:
            # Avanzar al siguiente paso
            self._active_sessions[session_id] = next_step
            return next_step.question, False

    def is_active(self, session_id: str) -> bool:
        """Devuelve True si hay un diagnóstico en curso para esa sesión."""
        return session_id in self._active_sessions

    def cancel(self, session_id: str) -> None:
        """Cancela el diagnóstico activo de una sesión."""
        self._active_sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normaliza la respuesta del usuario para emparejar con opciones."""
        text = text.lower().strip()
        # Mapear variantes comunes
        affirmatives = {"si", "sí", "yes", "claro", "correcto", "exacto", "afirmativo", "ok", "sip"}
        negatives = {"no", "nope", "negativo", "tampoco", "para nada"}
        if text in affirmatives:
            return "si"
        if text in negatives:
            return "no"
        return text

    @staticmethod
    def _match_option(key: str, options: dict) -> Optional[str]:
        """
        Busca la clave más cercana en las opciones disponibles.
        Las claves cortas ('si'/'no') requieren coincidencia exacta o que el
        texto del usuario esté compuesto íntegramente por esa palabra,
        evitando falsos positivos por subcadena (ej. 'no tengo internet' → 'no').
        """
        if key in options:
            return key
        # Para claves cortas (≤ 4 caracteres), solo coincidencia exacta
        for opt_key in options:
            if len(opt_key) <= 4:
                continue  # ya comprobado arriba
            if key in opt_key or opt_key in key:
                return opt_key
        return None
