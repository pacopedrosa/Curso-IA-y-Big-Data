"""
nlp_processor.py
----------------
Módulo de Procesamiento de Lenguaje Natural (PLN).
Realiza preprocesamiento de texto en español usando NLTK:
  - Normalización (minúsculas, eliminación de puntuación)
  - Tokenización
  - Eliminación de stopwords
  - Stemming con SnowballStemmer (español)
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Descargar recursos de NLTK necesarios (silencioso en producción)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


class NLPProcessor:
    """
    Procesador de texto en español que aplica un pipeline de PLN completo:
    normalización → tokenización → eliminación de stopwords → stemming.
    """

    def __init__(self, language: str = "spanish"):
        self.language = language
        self.stemmer = SnowballStemmer(language)
        # Stopwords base de NLTK en español
        self._stop_words = set(stopwords.words(language))
        # Partículas sin carga semántica en el dominio de soporte técnico
        self._stop_words.update({"favor", "porfavor"})

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """
        Devuelve el texto preprocesado listo para vectorización.
        Pasos: normalizar → tokenizar → filtrar → stemizar.
        """
        tokens = self._tokenize(self._normalize(text))
        tokens = [self.stemmer.stem(t) for t in tokens if self._is_valid(t)]
        return " ".join(tokens)

    def extract_keywords(self, text: str) -> list[str]:
        """
        Extrae las palabras clave (sin stopwords ni stemming) del texto.
        Útil para logging y análisis de consultas.
        """
        tokens = self._tokenize(self._normalize(text))
        return [t for t in tokens if self._is_valid(t)]

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """Convierte a minúsculas, elimina tildes y puntuación."""
        text = text.lower().strip()
        # Normalizar caracteres con tilde a su forma base
        replacements = str.maketrans("áéíóúüàèìòùâêîôûãñ", "aeiouuaeiouaeiouän")
        text = text.translate(replacements)
        # Eliminar puntuación
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Comprimir espacios múltiples
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        """Tokeniza el texto usando el tokenizador de NLTK."""
        try:
            return word_tokenize(text, language=self.language)
        except LookupError:
            # Fallback: tokenización simple por espacios
            return text.split()

    def _is_valid(self, token: str) -> bool:
        """Devuelve True si el token debe incluirse (alfabético y no es stopword)."""
        return token.isalpha() and token not in self._stop_words and len(token) > 2


# ---------------------------------------------------------------------------
# Procesador de PLN avanzado con spaCy
# ---------------------------------------------------------------------------

class SpacyProcessor:
    """
    Extractor de entidades técnicas y análisis morfológico usando spaCy.

    Complementa al pipeline NLTK/TF-IDF identificando entidades nombradas
    (productos, versiones, sistemas operativos, organizaciones) y extrayendo
    términos técnicos relevantes mediante análisis POS.

    Degradación elegante: si spaCy o el modelo no están disponibles,
    todos los métodos devuelven listas vacías sin lanzar excepción.

    Instalación del modelo:
        pip install spacy
        python -m spacy download es_core_news_sm
    """

    # Modelo de spaCy para español (pequeño, rápido, suficiente para NER)
    _MODEL = "es_core_news_sm"

    # Tipos de entidad relevantes en el dominio de soporte técnico
    _RELEVANT_LABELS = {"ORG", "PRODUCT", "MISC", "PER", "LOC"}

    # POS tags de tokens con carga semántica
    _CONTENT_POS = {"NOUN", "PROPN", "ADJ"}

    def __init__(self) -> None:
        self._nlp = None
        self._available = False
        self._load_model()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True si spaCy y el modelo están correctamente cargados."""
        return self._available

    def extract_entities(self, text: str) -> list[dict]:
        """
        Extrae entidades nombradas del texto.

        Retorna: lista de dicts {"text": str, "label": str} donde 'label'
        es el tipo de entidad spaCy (ORG, PRODUCT, MISC…).
        """
        if not self._available:
            return []
        doc = self._nlp(text)
        return [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in self._RELEVANT_LABELS
        ]

    def extract_technical_terms(self, text: str) -> list[str]:
        """
        Extrae sustantivos, nombres propios y adjetivos relevantes del texto
        en forma lematizada. Útil para enriquecer el contexto de diagnóstico.
        """
        if not self._available:
            return []
        doc = self._nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in self._CONTENT_POS
            and not token.is_stop
            and len(token.text) > 2
        ]

    def analyze(self, text: str) -> dict:
        """
        Análisis completo del texto: entidades + términos técnicos.

        Retorna:
            {
              "entities": [{"text": ..., "label": ...}, ...],
              "technical_terms": [...],
              "available": bool
            }
        """
        return {
            "entities": self.extract_entities(text),
            "technical_terms": self.extract_technical_terms(text),
            "available": self._available,
        }

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Intenta cargar el modelo spaCy en español. Falla silenciosamente."""
        try:
            import spacy  # noqa: PLC0415
            self._nlp = spacy.load(self._MODEL)
            self._available = True
        except ImportError:
            # spaCy no está instalado
            pass
        except OSError:
            # El modelo es_core_news_sm no está descargado
            # Instrucción: python -m spacy download es_core_news_sm
            pass
