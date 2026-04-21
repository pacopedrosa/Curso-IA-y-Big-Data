"""
Soporte de almacenamiento alternativo para SuperFresh:
  - PostgreSQL (relacional, recomendado para producción)
  - MongoDB    (documental, ideal para datos semiestructurados)

Uso:
    # PostgreSQL
    from database.storage import PostgreSQLStorage
    pg = PostgreSQLStorage(host="localhost", port=5432,
                           dbname="superfresh", user="postgres", password="secret")
    pg.export_from_sqlite()

    # MongoDB
    from database.storage import MongoDBStorage
    mg = MongoDBStorage(uri="mongodb://localhost:27017", db_name="superfresh")
    mg.export_from_sqlite()

Dependencias:
    pip install psycopg2-binary pymongo
"""
from __future__ import annotations

import os
import sqlite3
import pandas as pd

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "database", "superfresh.db")

TABLES = ["products", "stores", "promotions", "weather", "sales"]


# ── Helpers comunes ───────────────────────────────────────────────────────────
def _read_sqlite(table: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────
class PostgreSQLStorage:
    """
    Exporta los datos de SuperFresh desde SQLite a PostgreSQL.

    El esquema se crea automáticamente a partir de los DataFrames de Pandas
    usando pandas.DataFrame.to_sql() con el conector psycopg2.
    """

    def __init__(self, host: str = "localhost", port: int = 5432,
                 dbname: str = "superfresh", user: str = "postgres",
                 password: str = ""):
        try:
            import psycopg2  # noqa: F401 — solo verifica disponibilidad
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError(
                "Instala psycopg2 y sqlalchemy:\n"
                "  pip install psycopg2-binary sqlalchemy"
            )
        from sqlalchemy import create_engine
        conn_str = (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{host}:{port}/{dbname}"
        )
        self.engine = create_engine(conn_str)
        print(f"[PostgreSQL] Conectado a {host}:{port}/{dbname}")

    def export_from_sqlite(self, tables: list[str] = None, if_exists: str = "replace"):
        """
        Exporta las tablas indicadas desde SQLite a PostgreSQL.

        Args:
            tables:    Lista de tablas a exportar (None = todas).
            if_exists: Comportamiento si la tabla ya existe:
                       'replace' | 'append' | 'fail'
        """
        tables = tables or TABLES
        for table in tables:
            print(f"[PostgreSQL] Exportando tabla '{table}'…", end=" ")
            df = _read_sqlite(table)
            df.to_sql(table, self.engine, if_exists=if_exists, index=False,
                      method="multi", chunksize=5000)
            print(f"✅  {len(df):,} filas")

    def query(self, sql: str) -> pd.DataFrame:
        """Ejecuta una consulta SQL y devuelve un DataFrame."""
        return pd.read_sql(sql, self.engine)

    def get_monthly_revenue(self) -> pd.DataFrame:
        """Ejemplo de consulta analítica sobre PostgreSQL."""
        return self.query("""
            SELECT TO_CHAR(sale_date, 'YYYY-MM') AS year_month,
                   ROUND(SUM(revenue)::numeric, 2) AS monthly_revenue,
                   SUM(units_sold)                 AS monthly_units
            FROM   sales
            GROUP  BY TO_CHAR(sale_date, 'YYYY-MM')
            ORDER  BY year_month
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────────────────────────────────────
class MongoDBStorage:
    """
    Exporta los datos de SuperFresh desde SQLite a MongoDB.

    Cada tabla se almacena como una colección. Los registros de ventas
    incluyen el nombre del producto embebido para facilitar consultas.
    """

    def __init__(self, uri: str = "mongodb://localhost:27017",
                 db_name: str = "superfresh"):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "Instala pymongo:\n"
                "  pip install pymongo"
            )
        from pymongo import MongoClient
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        print(f"[MongoDB] Conectado → {uri}  BD: {db_name}")

    def export_from_sqlite(self, tables: list[str] = None,
                           drop_existing: bool = True):
        """
        Exporta las tablas indicadas a colecciones MongoDB.

        Las ventas se enriquecen con el nombre del producto antes
        de insertarlas (patrón de documento desnormalizado).
        """
        tables = tables or TABLES
        products_df = _read_sqlite("products")[["product_id", "name", "category"]]
        products_map = products_df.set_index("product_id").to_dict(orient="index")

        for table in tables:
            coll = self.db[table]
            if drop_existing:
                coll.drop()

            df = _read_sqlite(table)

            # Enriquecer ventas con datos de producto
            if table == "sales":
                df["product_name"] = df["product_id"].map(
                    lambda pid: products_map.get(pid, {}).get("name", "")
                )
                df["category"] = df["product_id"].map(
                    lambda pid: products_map.get(pid, {}).get("category", "")
                )

            # Convertir fechas a cadena para compatibilidad BSON
            for col in df.select_dtypes(include=["datetime64"]).columns:
                df[col] = df[col].dt.strftime("%Y-%m-%d")

            records = df.to_dict(orient="records")
            if records:
                coll.insert_many(records)
            print(f"[MongoDB] Colección '{table}' → {len(records):,} documentos ✅")

    def get_top_products(self, n: int = 10) -> list[dict]:
        """Agrega las ventas en MongoDB para obtener el top N de productos."""
        pipeline = [
            {"$group": {
                "_id": "$product_name",
                "total_revenue": {"$sum": "$revenue"},
                "total_units":   {"$sum": "$units_sold"},
            }},
            {"$sort": {"total_revenue": -1}},
            {"$limit": n},
        ]
        return list(self.db["sales"].aggregate(pipeline))

    def get_monthly_trend(self) -> list[dict]:
        """Calcula la tendencia mensual de ingresos desde MongoDB."""
        pipeline = [
            {"$addFields": {
                "year_month": {"$substr": ["$sale_date", 0, 7]}
            }},
            {"$group": {
                "_id": "$year_month",
                "monthly_revenue": {"$sum": "$revenue"},
                "monthly_units":   {"$sum": "$units_sold"},
            }},
            {"$sort": {"_id": 1}},
        ]
        return list(self.db["sales"].aggregate(pipeline))


# ── CLI de demostración ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SuperFresh — Exportación de datos")
    parser.add_argument("backend", choices=["postgresql", "mongodb"],
                        help="Backend de almacenamiento destino")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--dbname", default="superfresh")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="")
    parser.add_argument("--uri", default="mongodb://localhost:27017",
                        help="URI de conexión MongoDB")
    args = parser.parse_args()

    if args.backend == "postgresql":
        port = args.port or 5432
        pg = PostgreSQLStorage(host=args.host, port=port,
                               dbname=args.dbname, user=args.user,
                               password=args.password)
        pg.export_from_sqlite()
        print("\n[PostgreSQL] Tendencia mensual (primeras filas):")
        print(pg.get_monthly_revenue().head())

    elif args.backend == "mongodb":
        mg = MongoDBStorage(uri=args.uri, db_name=args.dbname)
        mg.export_from_sqlite()
        print("\n[MongoDB] Top 5 productos:")
        for doc in mg.get_top_products(5):
            print(f"  {doc['_id']:30s}  €{doc['total_revenue']:>10,.2f}")
