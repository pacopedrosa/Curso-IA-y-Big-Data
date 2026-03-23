from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import boto3
import sqlite3
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clientes AWS
rekognition = boto3.client("rekognition", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = os.getenv("S3_BUCKET_NAME")

# Base de datos
conn = sqlite3.connect("db.sqlite", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS imagenes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT,
        etiquetas TEXT,
        s3_url TEXT
    )
""")
conn.commit()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()

    # 1. Subir a S3
    key = f"productos/{uuid.uuid4()}_{file.filename}"
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=content,
        ContentType=file.content_type
    )
    s3_url = f"https://{BUCKET}.s3.us-east-1.amazonaws.com/{key}"

    # 2. Analizar con Rekognition
    response = rekognition.detect_labels(
        Image={"Bytes": content},
        MaxLabels=5,
        MinConfidence=99
    )
    labels = [label["Name"] for label in response["Labels"]]
    confianzas = {label["Name"]: round(label["Confidence"], 2) for label in response["Labels"]}
    etiquetas = ", ".join(labels)

    # 3. Guardar en base de datos
    cursor.execute(
        "INSERT INTO imagenes (nombre, etiquetas, s3_url) VALUES (?, ?, ?)",
        (file.filename, etiquetas, s3_url)
    )
    conn.commit()

    return {
        "filename": file.filename,
        "labels": labels,
        "confianzas": confianzas,
        "s3_url": s3_url
    }

@app.get("/imagenes")
def get_imagenes():
    cursor.execute("SELECT id, nombre, etiquetas, s3_url FROM imagenes ORDER BY id DESC")
    rows = cursor.fetchall()
    return [{"id": r[0], "nombre": r[1], "etiquetas": r[2], "s3_url": r[3]} for r in rows]

@app.get("/estadisticas")
def get_estadisticas():
    cursor.execute("SELECT etiquetas FROM imagenes")
    rows = cursor.fetchall()
    total_imagenes = len(rows)
    conteo = {}
    for (etiquetas,) in rows:
        for etiqueta in etiquetas.split(", "):
            etiqueta = etiqueta.strip()
            if etiqueta:
                conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
    top_etiquetas = sorted(conteo.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "total_imagenes_procesadas": total_imagenes,
        "etiquetas_mas_frecuentes": [{"etiqueta": k, "frecuencia": v} for k, v in top_etiquetas],
        "umbral_confianza_minima": 99,
        "max_etiquetas_por_imagen": 5
    }