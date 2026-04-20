"""
Script 3: predecir.py
Entrena un modelo de regresión lineal con los datos históricos almacenados
en Google Sheets y predice el precio medio de los libros para los próximos 7 días.
Las predicciones se guardan en la pestaña 'Predicciones' del mismo documento.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ── Configuración ──────────────────────────────────────────────────────────────
ARCHIVO_CREDENCIALES    = "credentials.json"
NOMBRE_HOJA_CALCULO     = "Precios_Libros"
NOMBRE_PESTANA_DATOS    = "Datos"
NOMBRE_PESTANA_PREDIC   = "Predicciones"
DIAS_PREDICCION         = 7
UMBRAL_ALERTA_PORCENTAJE = 10.0   # % de cambio que dispara una alerta en consola

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# ── Conexión Google Sheets ─────────────────────────────────────────────────────

def conectar_google_sheets() -> gspread.Client:
    if not os.path.exists(ARCHIVO_CREDENCIALES):
        sys.exit(
            f"ERROR: No se encontró '{ARCHIVO_CREDENCIALES}'.\n"
            "Consulta el README para obtener las credenciales de Google."
        )
    credenciales = Credentials.from_service_account_file(ARCHIVO_CREDENCIALES, scopes=SCOPES)
    return gspread.authorize(credenciales)


def obtener_o_crear_pestana(hoja: gspread.Spreadsheet, nombre: str) -> gspread.Worksheet:
    try:
        return hoja.worksheet(nombre)
    except gspread.WorksheetNotFound:
        return hoja.add_worksheet(title=nombre, rows=2000, cols=10)


# ── Carga y preparación de datos ───────────────────────────────────────────────

def cargar_datos_historicos(cliente: gspread.Client) -> pd.DataFrame:
    """Descarga los datos de la pestaña 'Datos' y los devuelve como DataFrame."""
    hoja = cliente.open(NOMBRE_HOJA_CALCULO)
    pestana = hoja.worksheet(NOMBRE_PESTANA_DATOS)
    registros = pestana.get_all_records()

    if not registros:
        sys.exit("ERROR: La hoja 'Datos' está vacía. Ejecuta primero extraer.py y data2google.py.")

    df = pd.DataFrame(registros)
    print(f"Datos cargados: {len(df)} registros.")
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y agrupa los datos por fecha, calculando el precio medio diario.
    Devuelve un DataFrame con columnas: fecha, precio_medio, dia_num.
    """
    # Asegurar tipos correctos
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")
    df["fecha_extraccion"] = pd.to_datetime(df["fecha_extraccion"], errors="coerce")
    df = df.dropna(subset=["precio", "fecha_extraccion"])

    # Agrupar por fecha (solo la parte de la fecha, sin hora)
    df["fecha"] = df["fecha_extraccion"].dt.date
    resumen = (
        df.groupby("fecha")["precio"]
        .mean()
        .reset_index()
        .rename(columns={"precio": "precio_medio"})
        .sort_values("fecha")
    )

    # Número de día secuencial (para la regresión)
    resumen["dia_num"] = range(len(resumen))

    print(f"Días únicos con datos: {len(resumen)}")
    print(resumen.to_string(index=False))
    return resumen


# ── Modelo de Machine Learning ─────────────────────────────────────────────────

def entrenar_modelo(resumen: pd.DataFrame) -> tuple[LinearRegression, float]:
    """Entrena una regresión lineal y devuelve el modelo junto al precio actual medio."""
    X = resumen[["dia_num"]].values
    y = resumen["precio_medio"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    # Métricas (solo informativas; con pocos datos pueden no ser representativas)
    y_pred = modelo.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred) if len(y) > 1 else float("nan")

    print(f"\nModelo entrenado — pendiente: {modelo.coef_[0]:.4f}, "
          f"intercepto: {modelo.intercept_:.4f}")
    print(f"MAE: {mae:.4f} | R²: {r2:.4f}")

    precio_actual = float(y[-1])
    return modelo, precio_actual


def generar_predicciones(
    modelo: LinearRegression,
    resumen: pd.DataFrame,
    precio_actual: float
) -> pd.DataFrame:
    """Genera predicciones para los próximos DIAS_PREDICCION días."""
    ultimo_dia_num = int(resumen["dia_num"].max())
    ultima_fecha   = pd.to_datetime(resumen["fecha"].max())

    predicciones = []
    for i in range(1, DIAS_PREDICCION + 1):
        dia_num        = ultimo_dia_num + i
        fecha_pred     = (ultima_fecha + timedelta(days=i)).date()
        precio_pred    = float(modelo.predict([[dia_num]])[0])
        cambio_pct     = ((precio_pred - precio_actual) / precio_actual) * 100

        predicciones.append({
            "fecha_prediccion": str(fecha_pred),
            "precio_predicho":  round(precio_pred, 4),
            "cambio_pct":       round(cambio_pct, 2),
            "alerta":           "SÍ" if abs(cambio_pct) >= UMBRAL_ALERTA_PORCENTAJE else "NO",
            "generado_en":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    df_pred = pd.DataFrame(predicciones)
    print("\nPredicciones generadas:")
    print(df_pred.to_string(index=False))

    # Alerta en consola
    alertas = df_pred[df_pred["alerta"] == "SÍ"]
    if not alertas.empty:
        print(f"\n⚠  ALERTA: Se detectaron {len(alertas)} días con cambio ≥ {UMBRAL_ALERTA_PORCENTAJE}%:")
        print(alertas[["fecha_prediccion", "precio_predicho", "cambio_pct"]].to_string(index=False))

    return df_pred


# ── Guardar predicciones en Google Sheets ─────────────────────────────────────

def guardar_predicciones(cliente: gspread.Client, df_pred: pd.DataFrame) -> None:
    """
    Añade las predicciones nuevas a la pestaña 'Predicciones', evitando duplicados
    por fecha de predicción.
    """
    hoja    = cliente.open(NOMBRE_HOJA_CALCULO)
    pestana = obtener_o_crear_pestana(hoja, NOMBRE_PESTANA_PREDIC)

    registros_existentes = pestana.get_all_records()

    if registros_existentes:
        fechas_existentes = {r["fecha_prediccion"] for r in registros_existentes}
    else:
        # Hoja vacía → escribir encabezados
        pestana.append_row(list(df_pred.columns))
        fechas_existentes = set()

    nuevas_filas = [
        row for row in df_pred.astype(str).values.tolist()
        if row[0] not in fechas_existentes
    ]

    if nuevas_filas:
        pestana.append_rows(nuevas_filas, value_input_option="USER_ENTERED")
        print(f"\n{len(nuevas_filas)} predicciones guardadas en '{NOMBRE_PESTANA_PREDIC}'.")
    else:
        print("\nNo hay predicciones nuevas (ya existen para esas fechas).")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== PREDICCIÓN DE PRECIOS ===\n")

    print("1. Conectando con Google Sheets …")
    cliente = conectar_google_sheets()

    print("\n2. Cargando datos históricos …")
    df_raw = cargar_datos_historicos(cliente)

    print("\n3. Preparando datos …")
    resumen = preparar_datos(df_raw)

    print("\n4. Entrenando modelo de regresión lineal …")
    modelo, precio_actual = entrenar_modelo(resumen)

    print("\n5. Generando predicciones para los próximos 7 días …")
    df_predicciones = generar_predicciones(modelo, resumen, precio_actual)

    print("\n6. Guardando predicciones en Google Sheets …")
    guardar_predicciones(cliente, df_predicciones)

    print("\n=== PROCESO COMPLETADO ===")


if __name__ == "__main__":
    main()
