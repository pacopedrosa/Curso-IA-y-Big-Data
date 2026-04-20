# Laboratorio 5 BD — Predicción de Precios en Marketplaces

Sistema completo de **web scraping + machine learning + cloud** para extraer precios de libros, almacenarlos en Google Sheets, y predecir tendencias futuras.

---

## Estructura del proyecto

```
lab Big Data/
├── extraer.py          # Script 1: scraping de books.toscrape.com → CSV
├── data2google.py      # Script 2: sube el CSV a Google Sheets
├── predecir.py         # Script 3: regresión lineal + predicción 7 días
├── app.py              # Servidor Flask para AWS EC2
├── requirements.txt    # Dependencias Python
├── credentials.json    # ← TÚ debes crear este archivo (ver paso 3)
└── data/
    └── precios_libros.csv   # generado automáticamente por extraer.py
```

---

## Fase 1: Ejecución en local

### Paso 1 — Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 2 — Ejecutar el scraping

```bash
python extraer.py
```

Esto visita todas las páginas de **https://books.toscrape.com**, extrae título, precio, rating y disponibilidad de cada libro, y guarda los resultados en `data/precios_libros.csv`.

---

### Paso 3 — Configurar Google Sheets

1. Ve a [Google Cloud Console](https://console.cloud.google.com/) y crea un proyecto.
2. Activa las APIs **Google Sheets API** y **Google Drive API**.
3. En *IAM y administración → Cuentas de servicio*, crea una cuenta y descarga la clave **JSON**.
4. Renombra ese archivo a **`credentials.json`** y colócalo en esta carpeta.
5. Crea un Google Sheet llamado exactamente **`Precios_Libros`**.
6. Comparte la hoja con el **email de la cuenta de servicio** (aparece en el JSON como `client_email`), dándole rol **Editor**.

### Paso 4 — Subir datos a Google Sheets

```bash
python data2google.py
```

Los datos del CSV se escriben en la pestaña **`Datos`** de la hoja `Precios_Libros`.

### Paso 5 — Generar predicciones

```bash
python predecir.py
```

El script:
- Lee los datos históricos de Google Sheets.
- Entrena un modelo de **regresión lineal** con scikit-learn.
- Predice el precio medio para los **próximos 7 días**.
- Guarda las predicciones en la pestaña **`Predicciones`** (sin duplicados).
- Muestra una **alerta** si alguna predicción supera el ±10 % respecto al precio actual.

---

## Fase 2: Cloud Computing con AWS EC2

### Paso 1 — Crear instancia EC2

1. En la consola AWS → EC2 → *Lanzar instancia*.
2. Selecciona **Windows Server 2022 Base** (para usar Chrome con Selenium).
3. Tipo de instancia: `t3.medium` o superior.
4. Grupos de seguridad: abrir puertos **RDP (3389)** y **HTTP (80)**.
5. Guarda el archivo `.pem` de la clave privada.

### Paso 2 — Conectar por RDP

1. En AWS, selecciona la instancia → *Conectar → Cliente RDP*.
2. Obtén la contraseña usando tu archivo `.pem`.
3. Abre **Conexión a Escritorio Remoto** con la IP pública, usuario `Administrator`.

### Paso 3 — Instalar Python y dependencias en la instancia

```powershell
# En la instancia Windows (PowerShell)
# 1. Descarga e instala Python desde https://python.org marcando "Add to PATH"
# 2. Instala dependencias:
pip install -r requirements.txt
```

### Paso 4 — Copiar archivos a la instancia

Copia todos los archivos del proyecto (incluido `credentials.json`) a la instancia mediante RDP (arrastrar y soltar en el escritorio remoto).

### Paso 5 — Arrancar el servidor Flask

```powershell
# Ejecutar en segundo plano con Gunicorn (o directamente):
python app.py
```

El servidor queda escuchando en `http://<IP_PUBLICA>/`.

**Endpoints disponibles:**

| Endpoint | Descripción |
|---|---|
| `GET /status` | Comprueba que el servidor está activo |
| `GET /extraer` | Ejecuta el scraping |
| `GET /descargar-csv` | Descarga el CSV generado |
| `GET /subir-google` | Sube el CSV a Google Sheets |
| `GET /predecir` | Ejecuta el modelo de predicción |

---

## Automatización con Make

1. Crea un escenario en [make.com](https://make.com).
2. **Módulo 1** — *HTTP: Make a request* → `GET http://<IP>/extraer` (programado cada 24 h).
3. **Módulo 2** — *HTTP: Make a request* → `GET http://<IP>/descargar-csv`.
4. **Módulo 3** — *CSV: Parse CSV* para estructurar los datos.
5. **Módulo 4** — *Google Sheets: Add a Row* para guardar en la hoja.
6. **Módulo 5** — *HTTP: Make a request* → `GET http://<IP>/predecir`.

---

## Tecnologías

| Tecnología | Uso |
|---|---|
| Python 3.11+ | Lenguaje principal |
| Selenium + webdriver-manager | Web scraping |
| pandas | Manipulación de datos |
| scikit-learn | Regresión lineal |
| gspread + google-auth | API de Google Sheets |
| Flask | Servidor web en AWS |
| AWS EC2 | Cloud computing |
| Make | Automatización |
