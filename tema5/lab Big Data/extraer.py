"""
Script 1: extraer.py
Extrae títulos, precios y metadatos de libros de https://books.toscrape.com
usando Selenium en modo headless y guarda los resultados en data/precios_libros.csv
"""

import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ── Configuración ──────────────────────────────────────────────────────────────
URL_BASE   = "https://books.toscrape.com/catalogue/"
URL_INICIO = "https://books.toscrape.com/catalogue/page-1.html"
ARCHIVO_CSV = "data/precios_libros.csv"
PAUSAS_SEG  = 1          # pausa entre páginas (evita sobrecargar el servidor)

# Mapa de valoración en texto → número de estrellas
RATING_MAP = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}


def configurar_driver() -> webdriver.Chrome:
    """Crea y devuelve un ChromeDriver en modo headless."""
    opciones = Options()
    opciones.add_argument("--headless")          # sin interfaz gráfica
    opciones.add_argument("--no-sandbox")
    opciones.add_argument("--disable-dev-shm-usage")
    opciones.add_argument("--disable-gpu")
    opciones.add_argument("--window-size=1920,1080")
    opciones.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    servicio = Service(ChromeDriverManager().install())
    driver   = webdriver.Chrome(service=servicio, options=opciones)
    return driver


def extraer_pagina(driver: webdriver.Chrome, url: str) -> list[dict]:
    """Navega a una URL y extrae los libros de esa página."""
    driver.get(url)
    # Esperar a que carguen los artículos
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.product_pod"))
    )

    libros = []
    articulos = driver.find_elements(By.CSS_SELECTOR, "article.product_pod")

    for articulo in articulos:
        # Título completo (está en el atributo 'title' del enlace <a> dentro de h3)
        titulo = articulo.find_element(By.CSS_SELECTOR, "h3 a").get_attribute("title")

        # Precio: eliminar símbolo '£' y convertir a float
        precio_texto = articulo.find_element(By.CSS_SELECTOR, "p.price_color").text
        precio = float(precio_texto.replace("£", "").replace("Â", "").strip())

        # Disponibilidad
        disponibilidad = articulo.find_element(By.CSS_SELECTOR, "p.availability").text.strip()

        # Rating (clase CSS: "star-rating One", "star-rating Two", etc.)
        rating_clase = articulo.find_element(By.CSS_SELECTOR, "p.star-rating").get_attribute("class")
        rating_texto = rating_clase.replace("star-rating", "").strip()
        rating = RATING_MAP.get(rating_texto, 0)

        libros.append({
            "titulo":        titulo,
            "precio":        precio,
            "rating":        rating,
            "disponibilidad": disponibilidad,
            "fecha_extraccion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    return libros


def obtener_siguiente_url(driver: webdriver.Chrome) -> str | None:
    """Devuelve la URL de la página siguiente, o None si es la última."""
    try:
        boton_sig = driver.find_element(By.CSS_SELECTOR, "li.next a")
        href = boton_sig.get_attribute("href")
        return href
    except Exception:
        return None


def main():
    print("Iniciando extracción de books.toscrape.com …")
    driver = configurar_driver()
    todos_los_libros = []

    url_actual = URL_INICIO
    pagina     = 1

    try:
        while url_actual:
            print(f"  Extrayendo página {pagina}: {url_actual}")
            libros = extraer_pagina(driver, url_actual)
            todos_los_libros.extend(libros)
            print(f"    → {len(libros)} libros obtenidos (total: {len(todos_los_libros)})")

            url_siguiente = obtener_siguiente_url(driver)
            if url_siguiente:
                # La URL devuelta puede ser relativa; asegurar URL absoluta
                if not url_siguiente.startswith("http"):
                    url_actual = URL_BASE + url_siguiente
                else:
                    url_actual = url_siguiente
                pagina += 1
                time.sleep(PAUSAS_SEG)
            else:
                break

    finally:
        driver.quit()

    # Guardar en CSV
    df = pd.DataFrame(todos_los_libros)
    df.to_csv(ARCHIVO_CSV, index=False, encoding="utf-8")
    print(f"\nExtracción completada. {len(df)} libros guardados en '{ARCHIVO_CSV}'")
    print(df.describe())


if __name__ == "__main__":
    main()
