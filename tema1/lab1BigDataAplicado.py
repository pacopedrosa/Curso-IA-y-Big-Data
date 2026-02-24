import pandas as pd
import sqlite3
import os

# Paso 1: Extraccion de los datos de el csv 
def extraer_datos(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe.")
    data = pd.read_csv(filepath)
    return data

# Paso 2: Transformacion de los datos

def transformar_datos(datos):
    print("Transformando datos...")
    datos['fecha'] = pd.to_datetime(datos['fecha'])
    datos['total'] = datos['precio']*datos['cantidad']
    datos_transformados = datos[['fecha', 'producto', 'cantidad', 'precio', 'total']]
    return datos_transformados

# Paso 3: Carga de los datos en una base de datos SQLite

def cargar_datos(datos, db_name='venta.db', table_name='ventas'):
    print("Cargando datos en la base de datos...")
    conn = sqlite3.connect(db_name)
    datos.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Datos cargados en la tabla '{table_name}' de la base de datos '{db_name}'.")
    
    # Visualizar datos cargados en SQLite
def visualizar_datos(db_name, query):
    """
    Ejecuta una consulta SQL en la base de datos SQLite y muestra los resultados.
    """
    conexion = sqlite3.connect(db_name)
    cursor = conexion.cursor()
    print(f"Ejecutando consulta: {query}\n")
    
    cursor.execute(query)
    resultados = cursor.fetchall()

    # Mostrar resultados en formato tabular
    print(f"{'ID':<5}{'Producto':<15}{'Categoría':<15}{'Fecha':<15}{'Total':<10}")
    print("-" * 60)
    for row in resultados:
        print(f"{row[0]:<5}{row[1]:<15}{row[2]:<15}{row[3]:<15}{row[4]:<10.2f}")

    conexion.close()
    
# Flujo pricipal del proceso ETL

def proceso_etl(filepath):
    print("INICIANDO PROCESO ETL...")
    datos = extraer_datos(filepath)
    print(datos.head())
    
    datos_transformados = transformar_datos(datos)
    print(datos_transformados.head())
    
    cargar_datos(datos_transformados)
    print("PROCESO ETL COMPLETADO.")
    

if __name__ == "__main__":
    archivo_csv = "ventas.csv"
    proceso_etl(archivo_csv)
    
    # Visualizar los datos cargados
    db_name = 'ventas.db'
    query = "SELECT * FROM ventas;"
    visualizar_datos(db_name, query)
    
    input("Presiona Enter para salir...")