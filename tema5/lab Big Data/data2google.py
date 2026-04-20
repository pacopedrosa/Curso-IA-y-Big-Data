"""
Script 2: data2google.py
Lee el CSV generado por extraer.py y sube los datos a Google Sheets.

Requisitos previos:
  1. Crear un proyecto en Google Cloud Console.
  2. Habilitar las APIs: Google Sheets API y Google Drive API.
  3. Crear una Cuenta de Servicio (Service Account) y descargar el archivo JSON de credenciales.
  4. Guardar el JSON como 'credentials.json' en esta misma carpeta.
  5. Crear un Google Sheet llamado 'Precios_Libros' y compartirlo (rol Editor) 
     con el email de la cuenta de servicio.
"""

import os
import sys
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ── Configuración ──────────────────────────────────────────────────────────────
ARCHIVO_CREDENCIALES = "credentials.json"   # ruta al JSON de la cuenta de servicio
NOMBRE_HOJA_CALCULO  = "Precios_Libros"     # nombre del Google Sheet
NOMBRE_PESTANA_DATOS = "Datos"              # pestaña donde se guardan los datos raw
ARCHIVO_CSV          = "data/precios_libros.csv"

# Ámbitos necesarios para leer/escribir en Sheets y Drive
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def conectar_google_sheets() -> gspread.Client:
    """Autenticarse con Google usando la cuenta de servicio y devolver el cliente."""
    if not os.path.exists(ARCHIVO_CREDENCIALES):
        sys.exit(
            f"ERROR: No se encontró el archivo de credenciales '{ARCHIVO_CREDENCIALES}'.\n"
            "Descárgalo desde Google Cloud Console → Cuentas de servicio → Claves."
        )
    credenciales = Credentials.from_service_account_file(ARCHIVO_CREDENCIALES, scopes=SCOPES)
    cliente = gspread.authorize(credenciales)
    return cliente


def leer_csv() -> pd.DataFrame:
    """Lee el CSV generado por extraer.py."""
    if not os.path.exists(ARCHIVO_CSV):
        sys.exit(
            f"ERROR: No se encontró '{ARCHIVO_CSV}'.\n"
            "Ejecuta primero el script extraer.py para generar los datos."
        )
    df = pd.read_csv(ARCHIVO_CSV)
    print(f"CSV cargado: {len(df)} registros.")
    return df


def obtener_o_crear_pestana(hoja_calculo: gspread.Spreadsheet, nombre: str) -> gspread.Worksheet:
    """Devuelve la pestaña con ese nombre; la crea si no existe."""
    try:
        return hoja_calculo.worksheet(nombre)
    except gspread.WorksheetNotFound:
        return hoja_calculo.add_worksheet(title=nombre, rows=2000, cols=20)


def subir_datos(cliente: gspread.Client, df: pd.DataFrame) -> None:
    """Sube el DataFrame a la hoja de cálculo de Google Sheets."""
    # Abrir el documento
    hoja_calculo = cliente.open(NOMBRE_HOJA_CALCULO)
    pestana = obtener_o_crear_pestana(hoja_calculo, NOMBRE_PESTANA_DATOS)

    # Limpiar contenido anterior
    pestana.clear()
    print(f"Pestaña '{NOMBRE_PESTANA_DATOS}' limpiada.")

    # Escribir encabezados
    encabezados = list(df.columns)
    pestana.append_row(encabezados)

    # Escribir filas en lotes para mayor eficiencia
    filas = df.astype(str).values.tolist()
    pestana.append_rows(filas, value_input_option="USER_ENTERED")

    print(f"{len(filas)} filas escritas en Google Sheets → '{NOMBRE_HOJA_CALCULO}' / '{NOMBRE_PESTANA_DATOS}'")


def main():
    print("Conectando con Google Sheets …")
    cliente = conectar_google_sheets()

    print("Leyendo CSV …")
    df = leer_csv()

    print("Subiendo datos …")
    subir_datos(cliente, df)

    print("\nProceso finalizado con éxito.")


if __name__ == "__main__":
    main()
