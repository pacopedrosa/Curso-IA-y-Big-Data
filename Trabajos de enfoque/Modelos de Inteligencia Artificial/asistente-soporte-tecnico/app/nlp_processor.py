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
