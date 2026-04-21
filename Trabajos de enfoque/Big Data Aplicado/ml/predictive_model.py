"""
predictive_model.py — Modelo predictivo de anomalías (SmartManuTech)
Algoritmo: KMeans (PySpark MLlib) — detección basada en distancia al centroide.

Entrenamiento con PySpark MLlib (SparkSession local), inferencia con numpy
para garantizar eficiencia en tiempo real.

Estrategia:
  - PySpark MLlib KMeans agrupa los datos históricos en k=2 clústeres.
  - El pipeline de Spark incluye VectorAssembler + StandardScaler + KMeans.
  - El umbral de anomalía se fija en el percentil 95 de distancias al centroide.
  - Los parámetros entrenados (centroides, media, std) se exportan a numpy
    para inferencia eficiente sin overhead de Spark en producción.
  - Reentrenamiento online cada 500 nuevas muestras (adaptación a concept drift).
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Campos de entrada al modelo
FEATURES = ["temperatura", "vibracion", "velocidad_rpm", "consumo_kw", "presion_bar"]


class DetectorAnomalias:
    """
    Detector de anomalías multivariante basado en KMeans de PySpark MLlib.

    Entrenamiento:
      - SparkSession local con PySpark MLlib Pipeline:
          VectorAssembler → StandardScaler → KMeans(k=2)
      - Los parámetros resultantes (centroides, media/std del scaler)
        se extraen y se usan en numpy para la inferencia en tiempo real.

    Inferencia:
      - Distancia euclidiana al centroide más cercano (numpy, sin Spark).
      - Si distancia > umbral (percentil 95), el punto se clasifica como anómalo.

    Ventajas sobre umbrales simples:
      - Detecta combinaciones anómalas multivariantes.
      - No requiere datos etiquetados (no supervisado).
      - PySpark MLlib permite escalar el entrenamiento a datasets distribuidos.
    """

    def __init__(self, k: int = 2, contamination: float = 0.05):
        self.k = k
        self.contamination = contamination
        self.centers: np.ndarray | None = None      # centroides exportados de Spark
        self.scaler_mean: np.ndarray | None = None  # media del StandardScaler de Spark
        self.scaler_std: np.ndarray | None = None   # std del StandardScaler de Spark
        self.threshold: float | None = None         # umbral de distancia para anomalía
        self.entrenado = False
        self.ultimo_score: float = 0.0
        self._buffer: list = []
        self.BUFFER_REENTRENAMIENTO = 500

        self._warm_up()

    def _warm_up(self):
        """
        Genera 2.000 registros históricos de operación normal simulada
        e inicia el entrenamiento con PySpark MLlib.
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

        self._entrenar_spark(datos)

    def _entrenar_spark(self, datos: np.ndarray):
        """
        Entrena el modelo KMeans usando PySpark MLlib.

        Pipeline de Spark:
          1. VectorAssembler: concatena los features en un vector denso.
          2. StandardScaler: normaliza media=0, std=1 por columna.
          3. KMeans(k=2): agrupa en clústeres (normal / anómalo).

        Los parámetros se exportan a numpy para la inferencia en tiempo real.
        """
        spark = None
        try:
            from pyspark.sql import SparkSession
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.feature import StandardScaler as SparkStandardScaler
            from pyspark.ml.clustering import KMeans
            from pyspark.ml import Pipeline

            spark = (
                SparkSession.builder
                .appName("SmartManuTech-AnomalyDetector")
                .master("local[*]")
                .config("spark.driver.memory", "512m")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
                .getOrCreate()
            )
            spark.sparkContext.setLogLevel("ERROR")

            # Crear DataFrame de Spark a partir de los datos numpy
            rows = [tuple(float(v) for v in row) for row in datos]
            df = spark.createDataFrame(rows, FEATURES)

            # Pipeline: VectorAssembler → StandardScaler → KMeans
            assembler = VectorAssembler(inputCols=FEATURES, outputCol="features_raw")
            scaler = SparkStandardScaler(
                inputCol="features_raw", outputCol="features",
                withMean=True, withStd=True
            )
            kmeans = KMeans(
                k=self.k, seed=42,
                featuresCol="features", predictionCol="cluster",
                maxIter=30
            )
            pipeline = Pipeline(stages=[assembler, scaler, kmeans])
            model = pipeline.fit(df)

            # Exportar parámetros del StandardScaler a numpy
            # En PySpark 3.5+ mean y std ya son numpy arrays (no DenseVector)
            scaler_model = model.stages[1]
            mean_raw = scaler_model.mean
            std_raw  = scaler_model.std
            self.scaler_mean = np.array(mean_raw.toArray() if hasattr(mean_raw, "toArray") else mean_raw)
            std_arr = np.array(std_raw.toArray() if hasattr(std_raw, "toArray") else std_raw)
            self.scaler_std = np.where(std_arr == 0, 1.0, std_arr)

            # Exportar centroides del KMeans a numpy
            kmeans_model = model.stages[2]
            centers_raw = kmeans_model.clusterCenters()
            self.centers = np.array([
                c.toArray() if hasattr(c, "toArray") else np.array(c)
                for c in centers_raw
            ])

            # Calcular umbral: percentil (1 - contamination)*100 de las distancias
            datos_scaled = (datos - self.scaler_mean) / self.scaler_std
            distancias = self._distancias_min_centroide(datos_scaled)
            self.threshold = float(np.percentile(distancias, (1 - self.contamination) * 100))

            self.entrenado = True
            logger.info(
                f"PySpark MLlib KMeans entrenado: k={self.k} | "
                f"muestras={len(datos)} | threshold={self.threshold:.4f} | "
                f"centroides={self.centers.shape}"
            )

        except Exception as e:
            logger.error(f"Error entrenando con PySpark MLlib: {e}")
            logger.warning("Activando fallback numpy para detección de anomalías")
            self._fallback_init(datos)
        finally:
            if spark:
                try:
                    spark.stop()
                except Exception:
                    pass

    def _fallback_init(self, datos: np.ndarray):
        """Fallback con numpy puro si PySpark no está disponible."""
        self.scaler_mean = datos.mean(axis=0)
        self.scaler_std = datos.std(axis=0)
        self.scaler_std = np.where(self.scaler_std == 0, 1.0, self.scaler_std)
        datos_scaled = (datos - self.scaler_mean) / self.scaler_std
        self.centers = np.array([datos_scaled.mean(axis=0)])
        distancias = self._distancias_min_centroide(datos_scaled)
        self.threshold = float(np.percentile(distancias, (1 - self.contamination) * 100))
        self.entrenado = True
        logger.warning("Fallback numpy activo (sin PySpark)")

    def _distancias_min_centroide(self, X: np.ndarray) -> np.ndarray:
        """Distancia euclidiana mínima a cualquier centroide."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in self.centers])
        return dists.min(axis=0)

    def _extraer_features(self, dato: dict) -> np.ndarray:
        return np.array([
            float(dato.get("temperatura",   75.0)),
            float(dato.get("vibracion",      2.5)),
            float(dato.get("velocidad_rpm", 1500)),
            float(dato.get("consumo_kw",    45.0)),
            float(dato.get("presion_bar",    6.0)),
        ])

    def predecir(self, dato: dict) -> bool:
        """
        Predice si un dato es anómalo usando los parámetros exportados de PySpark MLlib.
        - Escala el punto con los parámetros del StandardScaler de Spark (numpy).
        - Calcula la distancia euclidiana al centroide KMeans más cercano (numpy).
        - True si distancia > umbral (percentil 95 del histórico de entrenamiento).
        """
        if not self.entrenado:
            return False
        try:
            x = self._extraer_features(dato)
            x_scaled = (x - self.scaler_mean) / self.scaler_std
            dist = float(self._distancias_min_centroide(x_scaled)[0])
            self.ultimo_score = dist

            # Acumular para reentrenamiento online
            self._buffer.append(x.tolist())
            if len(self._buffer) >= self.BUFFER_REENTRENAMIENTO:
                self._reentrenar()

            return dist > self.threshold
        except Exception as e:
            logger.error(f"Error en predicción ML: {e}")
            return False

    def score_anomalia(self, dato: dict) -> float:
        """Distancia euclidiana al centroide más cercano (mayor = más anómalo)."""
        if not self.entrenado:
            return 0.0
        try:
            x = self._extraer_features(dato)
            x_scaled = (x - self.scaler_mean) / self.scaler_std
            return float(self._distancias_min_centroide(x_scaled)[0])
        except Exception:
            return 0.0

    def _reentrenar(self):
        """
        Reentrenamiento online con PySpark MLlib cada BUFFER_REENTRENAMIENTO muestras.
        Permite adaptar el modelo a derivas del proceso (concept drift).
        """
        try:
            datos = np.array(self._buffer)
            logger.info(f"Iniciando reentrenamiento PySpark MLlib con {len(datos)} muestras...")
            self._entrenar_spark(datos)
            self._buffer = []
        except Exception as e:
            logger.error(f"Error en reentrenamiento: {e}")
            self._buffer = []
