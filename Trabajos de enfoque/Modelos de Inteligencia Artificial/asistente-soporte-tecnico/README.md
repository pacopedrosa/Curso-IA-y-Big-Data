# Asistente Virtual de Soporte Técnico

Asistente conversacional en Python para el **soporte técnico de una empresa de servicios tecnológicos**. Clasifica automáticamente las consultas de los usuarios usando **Procesamiento de Lenguaje Natural (PLN)** y ofrece soluciones guiadas en tiempo real.

> Trabajo de Enfoque — Módulo: Modelos de Inteligencia Artificial  
> CE de Inteligencia Artificial y Big Data — 2025/2026

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────┐
│              Interfaz Web (HTML/JS)          │
└──────────────────┬──────────────────────────┘
                   │ HTTP (REST)
┌──────────────────▼──────────────────────────┐
│           API REST (FastAPI)                 │
│   POST /api/chat   GET /api/faqs             │
│   GET  /api/health GET /api/history          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│               Chatbot (orquestador)          │
│                                              │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  NLP Processor  │  │ Diagnostic Engine │  │
│  │  (NLTK/Stemmer) │  │  (flujos guiados) │  │
│  └────────┬────────┘  └──────────────────┘  │
│           │                                  │
│  ┌────────▼────────────────┐                 │
│  │   Intent Classifier     │                 │
│  │  TF-IDF + Logistic Reg. │                 │
│  └─────────────────────────┘                 │
└──────────────────────────────────────────────┘
```

## Tecnologías utilizadas

| Componente         | Tecnología                       |
|--------------------|----------------------------------|
| Lenguaje           | Python 3.11+                     |
| API REST           | FastAPI + Uvicorn                |
| PLN                | NLTK (tokenización, stopwords, stemming) |
| Clasificación      | scikit-learn (TF-IDF + Logistic Regression) |
| Interfaz web       | HTML5 / CSS3 / JavaScript vanilla |
| Tests              | pytest                           |

## Estructura del proyecto

```
asistente-soporte-tecnico/
├── app/
│   ├── __init__.py
│   ├── main.py              # API FastAPI
│   ├── chatbot.py           # Orquestador principal
│   ├── nlp_processor.py     # Pipeline de PLN
│   ├── intent_classifier.py # Clasificador TF-IDF + LogReg
│   ├── diagnostics.py       # Motor de diagnóstico guiado
│   └── data/
│       ├── intents.json     # Dataset de entrenamiento (15 intenciones)
│       └── model.pkl        # Modelo serializado (se genera al entrenar)
├── static/
│   └── index.html           # Interfaz web del chatbot
├── tests/
│   ├── __init__.py
│   └── test_chatbot.py      # Suite de tests (pytest)
├── train.py                 # Script de entrenamiento
├── requirements.txt
└── README.md
```

## Instalación y puesta en marcha

### 1. Clonar el repositorio

```bash
git clone https://github.com/pacopedrosa/asistente-soporte-tecnico.git
cd asistente-soporte-tecnico
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Entrenar el modelo

```bash
python train.py
```

Salida esperada:
```
✅ Entrenamiento completado en 0.43s
   Intenciones: 15
   Muestras de entrenamiento: 194
   Validación cruzada (5-fold): Accuracy media: 0.964 ± 0.021
```

### 4. Iniciar la API

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9050
```

### 5. Abrir el chatbot

Abre en tu navegador: [http://localhost:8000](http://localhost:8000)

O usa la API directamente:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "usuario_1", "message": "no tengo internet"}'
```

## Endpoints de la API

| Método   | Endpoint                          | Descripción                         |
|----------|-----------------------------------|-------------------------------------|
| `GET`    | `/`                               | Interfaz web del chatbot            |
| `POST`   | `/api/chat`                       | Enviar mensaje al asistente         |
| `GET`    | `/api/health`                     | Estado del servicio                 |
| `GET`    | `/api/faqs`                       | Listado de temas disponibles        |
| `GET`    | `/api/history/{session_id}`       | Historial de conversación           |
| `DELETE` | `/api/session/{session_id}`       | Limpiar sesión                      |

Documentación interactiva (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

## Intenciones soportadas

| Intención             | Descripción                              |
|-----------------------|------------------------------------------|
| `saludo`              | Saludos iniciales                        |
| `problema_wifi`       | Problemas de conexión a internet/WiFi    |
| `computadora_lenta`   | Equipo lento o con bajo rendimiento      |
| `reinicio_contrasena` | Recuperación y cambio de contraseñas     |
| `instalar_software`   | Instalación de programas                 |
| `no_enciende`         | El equipo no arranca o pantalla negra    |
| `virus_malware`       | Infecciones de malware o ransomware      |
| `problema_impresora`  | Impresora sin conexión o con errores     |
| `backup_datos`        | Copias de seguridad y recuperación       |
| `actualizaciones`     | Problemas con Windows Update o drivers   |
| `correo_electronico`  | Problemas con clientes de correo         |
| `bateria_portatil`    | Batería del portátil no carga            |
| `audio_video`         | Problemas de sonido o pantalla           |
| `soporte_humano`      | Derivar a técnico especialista           |
| `agradecimiento`      | Expresiones de agradecimiento            |

## Ejecutar los tests

```bash
pytest tests/ -v
```

## Consideraciones éticas y de privacidad

- **Privacidad por diseño**: no se persiste ningún dato de conversación en base de datos.
- Las sesiones se almacenan únicamente en memoria volátil y se eliminan al cerrar el servidor.
- No se recopilan datos personales del usuario más allá del contenido de la consulta.
- En producción, se debe restringir el CORS a dominios autorizados y añadir autenticación.
- El sistema es transparente: siempre informa al usuario cuando la consulta está fuera de su capacidad y ofrece alternativas de contacto humano.

## Licencia

MIT © 2026
