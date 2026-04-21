"""
Procesamiento de grandes volúmenes de datos con Apache Spark — SuperFresh.

Este módulo demuestra cómo usar PySpark para:
  1. Cargar y transformar los datos de ventas (ETL).
  2. Calcular KPIs agregados a escala (por producto, tienda y mes).
  3. Detectar tendencias estacionales mediante ventanas temporales.
  4. Exportar los resultados para ser consumidos por los modelos de IA.

Requisito: pip install pyspark
Nota: PySpark requiere Java 8+ instalado en el sistema.
"""
from __future__ import annotations

import os
import sys
import sqlite3
import pandas as pd

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH  = os.path.join(ROOT, "database", "superfresh.db")
DATA_DIR = os.path.dirname(__file__)
SPARK_OUT = os.path.join(ROOT, "database", "spark_output")


# ── Sesión Spark ──────────────────────────────────────────────────────────────
def get_spark_session(app_name: str = "SuperFresh-BigData"):
    """Crea y devuelve una sesión Spark configurada para uso local."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "PySpark no está instalado.\n"
            "Instálalo con: pip install pyspark\n"
            "También necesitas Java 8+ en tu sistema."
        )

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                           # Usa todos los cores locales
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")  # Ajustado a datos locales
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ── Exportar datos desde SQLite a CSV para Spark ──────────────────────────────
def export_sqlite_to_csv() -> dict[str, str]:
    """
    Exporta las tablas SQLite a CSV para que Spark pueda leerlas.
    Retorna un diccionario {tabla: ruta_csv}.
    """
    conn = sqlite3.connect(DB_PATH)
    paths = {}
    for table in ["sales", "products", "stores", "weather", "promotions"]:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            path = os.path.join(DATA_DIR, f"{table}.csv")
            df.to_csv(path, index=False)
            paths[table] = path
            print(f"[Spark-Export] {table} → {path}  ({len(df):,} filas)")
        except Exception as e:
            print(f"[Spark-Export] Tabla '{table}' omitida: {e}")
    conn.close()
    return paths


# ── ETL con Spark ─────────────────────────────────────────────────────────────
def load_spark_dataframes(spark, paths: dict):
    """Carga los CSV en DataFrames Spark con inferencia de esquema."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType

    df_sales = (spark.read
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(paths["sales"])
                .withColumn("sale_date", F.to_date("sale_date"))
                .withColumn("units_sold", F.col("units_sold").cast(IntegerType()))
                .withColumn("revenue", F.col("revenue").cast(DoubleType())))

    df_products = (spark.read
                   .option("header", "true")
                   .option("inferSchema", "true")
                   .csv(paths["products"]))

    df_stores = (spark.read
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .csv(paths["stores"]))

    return df_sales, df_products, df_stores


# ── Análisis KPI con Spark SQL ────────────────────────────────────────────────
def run_kpi_analysis(spark, df_sales, df_products, df_stores):
    """
    Calcula KPIs de negocio usando operaciones distribuidas de Spark:
      - Top 10 productos por ingresos totales.
      - Tendencia mensual de ingresos.
      - Ingresos por tienda.
      - Análisis de estacionalidad por categoría y mes.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Registrar vistas temporales para usar Spark SQL
    df_sales.createOrReplaceTempView("sales")
    df_products.createOrReplaceTempView("products")
    df_stores.createOrReplaceTempView("stores")

    print("\n" + "=" * 60)
    print("  SuperFresh — Análisis Big Data con Apache Spark")
    print("=" * 60)
    print(f"\n[Spark] Total de registros de ventas: {df_sales.count():,}")

    # ── 1. Top 10 productos por ingresos ──────────────────────────────────────
    print("\n[Spark] ── Top 10 productos por ingresos totales ──")
    top_products = spark.sql("""
        SELECT p.name, p.category,
               SUM(s.units_sold)  AS total_units,
               ROUND(SUM(s.revenue), 2) AS total_revenue
        FROM   sales s
        JOIN   products p ON s.product_id = p.product_id
        GROUP  BY p.name, p.category
        ORDER  BY total_revenue DESC
        LIMIT  10
    """)
    top_products.show(truncate=False)

    # ── 2. Tendencia mensual ──────────────────────────────────────────────────
    print("[Spark] ── Tendencia mensual de ingresos ──")
    monthly_trend = spark.sql("""
        SELECT DATE_FORMAT(sale_date, 'yyyy-MM') AS year_month,
               ROUND(SUM(revenue), 2)            AS monthly_revenue,
               SUM(units_sold)                   AS monthly_units
        FROM   sales
        GROUP  BY DATE_FORMAT(sale_date, 'yyyy-MM')
        ORDER  BY year_month
    """)
    monthly_trend.show(36, truncate=False)

    # ── 3. Ingresos por tienda ────────────────────────────────────────────────
    print("[Spark] ── Ingresos por tienda ──")
    by_store = spark.sql("""
        SELECT st.name AS store_name, st.city,
               ROUND(SUM(s.revenue), 2) AS total_revenue,
               SUM(s.units_sold)        AS total_units
        FROM   sales s
        JOIN   stores st ON s.store_id = st.store_id
        GROUP  BY st.name, st.city
        ORDER  BY total_revenue DESC
    """)
    by_store.show(truncate=False)

    # ── 4. Estacionalidad por categoría y mes ─────────────────────────────────
    print("[Spark] ── Estacionalidad: unidades medias por categoría y mes ──")
    seasonality = spark.sql("""
        SELECT p.category,
               MONTH(s.sale_date) AS month,
               ROUND(AVG(s.units_sold), 2) AS avg_units
        FROM   sales s
        JOIN   products p ON s.product_id = p.product_id
        GROUP  BY p.category, MONTH(s.sale_date)
        ORDER  BY p.category, month
    """)
    seasonality.show(120, truncate=False)

    # ── 5. Impacto del descuento con ventana deslizante ───────────────────────
    print("[Spark] ── Impacto del descuento en ventas (ventana 7 días) ──")
    window_spec = Window.partitionBy("product_id", "store_id").orderBy("sale_date") \
                        .rowsBetween(-3, 3)
    df_windowed = (df_sales
                   .withColumn("rolling_avg_7d", F.avg("units_sold").over(window_spec))
                   .groupBy("discount_pct")
                   .agg(F.round(F.avg("rolling_avg_7d"), 2).alias("avg_units_7d_window"))
                   .orderBy("discount_pct"))
    df_windowed.show(20, truncate=False)

    return top_products, monthly_trend, by_store, seasonality


# ── Guardar resultados Spark ──────────────────────────────────────────────────
def save_spark_results(top_products, monthly_trend, by_store, seasonality):
    """Guarda los DataFrames Spark como CSV para consumo posterior."""
    os.makedirs(SPARK_OUT, exist_ok=True)

    # Convertir a Pandas y guardar (pequeño volumen)
    for name, sdf in [("top_products", top_products),
                      ("monthly_trend", monthly_trend),
                      ("by_store", by_store),
                      ("seasonality", seasonality)]:
        path = os.path.join(SPARK_OUT, f"{name}.csv")
        sdf.toPandas().to_csv(path, index=False)
        print(f"[Spark] Resultado guardado → {path}")


# ── Pipeline Spark completo ───────────────────────────────────────────────────
def run_spark_pipeline():
    """
    Ejecuta el pipeline completo de Big Data con Spark:
      1. Exportar datos SQLite → CSV
      2. Cargar en Spark
      3. Calcular KPIs
      4. Guardar resultados
    """
    print("[Spark] Exportando datos desde SQLite…")
    paths = export_sqlite_to_csv()

    if "sales" not in paths:
        raise RuntimeError("No se pudieron exportar los datos. Ejecuta init_db.py primero.")

    print("[Spark] Iniciando sesión Apache Spark…")
    spark = get_spark_session()

    print("[Spark] Cargando DataFrames…")
    df_sales, df_products, df_stores = load_spark_dataframes(spark, paths)

    print("[Spark] Ejecutando análisis KPI…")
    top_products, monthly_trend, by_store, seasonality = run_kpi_analysis(
        spark, df_sales, df_products, df_stores
    )

    print("[Spark] Guardando resultados…")
    save_spark_results(top_products, monthly_trend, by_store, seasonality)

    spark.stop()
    print("\n[Spark] ✅ Pipeline Spark completado.")


if __name__ == "__main__":
    run_spark_pipeline()
