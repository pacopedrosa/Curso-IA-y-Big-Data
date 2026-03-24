"""
predictive_model.py — Modelo predictivo de anomalías (SmartManuTech)
Algoritmo: Isolation Forest (detección no supervisada, sin etiquetas)
Entrenamiento inicial con 2.000 muestras históricas simuladas de operación normal.
Reentrenamiento online cada 500 nuevas muestras recibidas.
"""
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Campos de entrada al modelo
FEATURES = ["temperatura", "vibracion", "velocidad_rpm", "consumo_kw", "presion_bar"]


class DetectorAnomalias:
    """
    Detector de anomalías multivariante basado en Isolation Forest.

    Isolation Forest construye árboles binarios de partición aleatoria.
    Los puntos que se aíslan rápidamente (pocas particiones) son anómalos.
    - contamination=0.05 → estima que el 5% de los datos son anomalías
    - n_estimators=100   → 100 árboles de decisión en el ensemble

    Ventajas sobre umbrales simples:
    - Detecta combinaciones anómalas (ej: temperatura normal + vibración alta)
    - No requiere datos etiquetados
    - Escalable y rápido en producción
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.modelo = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.entrenado = False
        self.ultimo_score: float = 0.0
        self._buffer: list = []
        self.BUFFER_REENTRENAMIENTO = 500

        self._warm_up()

    def _warm_up(self):
        """
        Simula 2.000 registros históricos de operación normal.
        En producción real, estos vendrían de la base de datos histórica.
        Los parámetros (media, std) corresponden a los rangos normales de las 5 máquinas.
        """
        np.random.seed(42)
        n = 2000

        datos = np.column_stack([
            np.random.normal(75,  4,   n),   # temperatura (°C)
            np.random.normal(2.5, 0.3, n),   # vibración (mm/s)
            np.random.normal(1500, 80, n),   # velocidad_rpm
            np.random.normal(45,  5,   n),   # consumo_kw
            np.random.normal(6.0, 0.4, n),   # presion_bar
        ])

        # Añadir ~5% de anomalías al histórico para que el modelo aprenda el contraste
        idx = np.random.choice(n, int(n * 0.05), replace=False)
        datos[idx, 0] += np.random.uniform(25, 45, len(idx))   # sobrecalentamiento
        datos[idx, 1] += np.random.uniform(2.0, 5.0, len(idx)) # vibración alta

        self.scaler.fit(datos)
        self.modelo.fit(self.scaler.transform(datos))
        self.entrenado = True
        logger.info(
            f"Isolation Forest entrenado: {n} muestras históricas | "
            f"contamination={self.modelo.contamination} | "
            f"n_estimators={self.modelo.n_estimators}"
        )

    def _extraer_features(self, dato: dict) -> np.ndarray:
        return np.array([[
            float(dato.get("temperatura",   75.0)),
            float(dato.get("vibracion",      2.5)),
            float(dato.get("velocidad_rpm", 1500)),
            float(dato.get("consumo_kw",    45.0)),
            float(dato.get("presion_bar",    6.0)),
        ]])

    def predecir(self, dato: dict) -> bool:
        """
        Predice si un dato es anómalo.
        Retorna True si el modelo considera el punto anómalo.
        Isolation Forest devuelve -1 para anomalías y +1 para datos normales.
        """
        if not self.entrenado:
            return False
        try:
            X = self.scaler.transform(self._extraer_features(dato))
            prediccion = self.modelo.predict(X)[0]
            self.ultimo_score = float(self.modelo.score_samples(X)[0])

            # Acumular para reentrenamiento online
            self._buffer.append(self._extraer_features(dato)[0].tolist())
            if len(self._buffer) >= self.BUFFER_REENTRENAMIENTO:
                self._reentrenar()

            return prediccion == -1
        except Exception as e:
            logger.error(f"Error en predicción ML: {e}")
            return False

    def score_anomalia(self, dato: dict) -> float:
        """Retorna el score de anomalía (más negativo = más anómalo, rango aprox. -1 a 0)."""
        if not self.entrenado:
            return 0.0
        try:
            X = self.scaler.transform(self._extraer_features(dato))
            return float(self.modelo.score_samples(X)[0])
        except Exception:
            return 0.0

    def _reentrenar(self):
        """
        Reentrenamiento incremental con muestras acumuladas.
        Permite adaptar el modelo a derivas del proceso (concept drift).
        """
        try:
            nuevos = np.array(self._buffer)
            self.modelo.fit(self.scaler.transform(nuevos))
            logger.info(f"Modelo reentrenado con {len(self._buffer)} nuevas muestras")
            self._buffer = []
        except Exception as e:
            logger.error(f"Error en reentrenamiento: {e}")
