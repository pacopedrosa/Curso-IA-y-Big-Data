"""
Generación de datos sintéticos para el sistema de recomendación de cursos.
Simula una plataforma educativa con usuarios, cursos e interacciones.
"""
import numpy as np
import pandas as pd
import random
import sqlite3
import os

# Reproducibilidad
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Constantes ────────────────────────────────────────────────────────────────
N_USERS = 500
N_COURSES = 50

CATEGORIES = ["Programación", "Data Science", "DevOps", "Diseño", "Negocios", "Idiomas"]

COURSES_LIST = [
    ("Python para principiantes",     "Programación", "básico",       4.5),
    ("Python avanzado",               "Programación", "avanzado",     4.7),
    ("JavaScript moderno",            "Programación", "intermedio",   4.3),
    ("TypeScript desde cero",         "Programación", "básico",       4.2),
    ("React.js completo",             "Programación", "intermedio",   4.6),
    ("Node.js y Express",             "Programación", "intermedio",   4.4),
    ("Java para backend",             "Programación", "intermedio",   4.1),
    ("Algoritmos y estructuras",      "Programación", "avanzado",     4.8),
    ("SQL desde cero",                "Data Science", "básico",       4.3),
    ("PostgreSQL avanzado",           "Data Science", "avanzado",     4.5),
    ("Introducción a ML",             "Data Science", "básico",       4.6),
    ("Machine Learning avanzado",     "Data Science", "avanzado",     4.9),
    ("Deep Learning con PyTorch",     "Data Science", "avanzado",     4.8),
    ("Pandas y NumPy",                "Data Science", "básico",       4.4),
    ("Visualización con Matplotlib",  "Data Science", "básico",       4.2),
    ("Power BI para analistas",       "Data Science", "intermedio",   4.3),
    ("Estadística para DS",           "Data Science", "intermedio",   4.5),
    ("NLP con Python",                "Data Science", "avanzado",     4.7),
    ("Computer Vision",               "Data Science", "avanzado",     4.6),
    ("Big Data con Spark",            "Data Science", "avanzado",     4.5),
    ("Docker para devs",              "DevOps",       "básico",       4.4),
    ("Kubernetes en producción",      "DevOps",       "avanzado",     4.6),
    ("CI/CD con GitHub Actions",      "DevOps",       "intermedio",   4.3),
    ("Terraform y AWS",               "DevOps",       "avanzado",     4.5),
    ("Linux para desarrolladores",    "DevOps",       "básico",       4.2),
    ("Monitorización con Grafana",    "DevOps",       "intermedio",   4.1),
    ("Seguridad en aplicaciones",     "DevOps",       "avanzado",     4.7),
    ("Microservicios con FastAPI",    "DevOps",       "intermedio",   4.5),
    ("UI/UX Fundamentos",             "Diseño",       "básico",       4.3),
    ("Figma profesional",             "Diseño",       "intermedio",   4.4),
    ("Diseño responsive",             "Diseño",       "básico",       4.2),
    ("Motion Design",                 "Diseño",       "avanzado",     4.3),
    ("Branding digital",              "Diseño",       "intermedio",   4.1),
    ("Photoshop avanzado",            "Diseño",       "avanzado",     4.0),
    ("Ilustración vectorial",         "Diseño",       "intermedio",   4.2),
    ("Marketing digital",             "Negocios",     "básico",       4.3),
    ("SEO y SEM",                     "Negocios",     "intermedio",   4.4),
    ("Google Analytics 4",            "Negocios",     "básico",       4.2),
    ("Gestión ágil con Scrum",        "Negocios",     "básico",       4.5),
    ("Finanzas para emprendedores",   "Negocios",     "básico",       4.1),
    ("Liderazgo y equipos remotos",   "Negocios",     "intermedio",   4.3),
    ("Product Management",            "Negocios",     "avanzado",     4.6),
    ("Copywriting efectivo",          "Negocios",     "básico",       4.2),
    ("Inglés para TI",                "Idiomas",      "básico",       4.5),
    ("Inglés avanzado",               "Idiomas",      "avanzado",     4.4),
    ("Inglés para negocios",          "Idiomas",      "intermedio",   4.3),
    ("Alemán básico",                 "Idiomas",      "básico",       4.1),
    ("Francés intermedio",            "Idiomas",      "intermedio",   4.2),
    ("Chino mandarín básico",         "Idiomas",      "básico",       4.0),
    ("Portugués para negocios",       "Idiomas",      "básico",       4.1),
]

# Perfiles de usuario (determinan sus preferencias de categorías)
USER_PROFILES = {
    "developer":        {"Programación": 0.45, "DevOps": 0.25, "Data Science": 0.20, "Idiomas": 0.05, "Diseño": 0.03, "Negocios": 0.02},
    "data_scientist":   {"Data Science": 0.55, "Programación": 0.25, "Negocios": 0.10, "DevOps": 0.05, "Diseño": 0.03, "Idiomas": 0.02},
    "designer":         {"Diseño": 0.50, "Programación": 0.20, "Negocios": 0.15, "Idiomas": 0.10, "Data Science": 0.03, "DevOps": 0.02},
    "business":         {"Negocios": 0.50, "Idiomas": 0.20, "Diseño": 0.15, "Programación": 0.10, "Data Science": 0.03, "DevOps": 0.02},
    "devops_sre":       {"DevOps": 0.50, "Programación": 0.25, "Data Science": 0.15, "Negocios": 0.05, "Diseño": 0.03, "Idiomas": 0.02},
    "student":          {"Programación": 0.30, "Data Science": 0.25, "Idiomas": 0.20, "Negocios": 0.15, "Diseño": 0.05, "DevOps": 0.05},
}


def generate_users(n: int = N_USERS) -> pd.DataFrame:
    profile_names = list(USER_PROFILES.keys())
    profiles = np.random.choice(profile_names, size=n, p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05])
    ages = np.random.randint(18, 55, size=n)
    seniority = np.random.choice(["junior", "mid", "senior"], size=n, p=[0.40, 0.35, 0.25])
    return pd.DataFrame({
        "user_id":   [f"U{i:04d}" for i in range(1, n + 1)],
        "age":       ages,
        "profile":   profiles,
        "seniority": seniority,
    })


def generate_courses(courses: list = COURSES_LIST) -> pd.DataFrame:
    rows = []
    for idx, (name, category, level, avg_rating) in enumerate(courses, start=1):
        duration_map = {"básico": (5, 15), "intermedio": (15, 35), "avanzado": (30, 60)}
        lo, hi = duration_map[level]
        rows.append({
            "course_id":  f"C{idx:03d}",
            "name":       name,
            "category":   category,
            "level":      level,
            "avg_rating": avg_rating,
            "duration_h": round(random.uniform(lo, hi), 1),
        })
    return pd.DataFrame(rows)


def generate_interactions(users: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    """
    Genera interacciones usuario-curso con rating, porcentaje de progreso y
    duración de sesión, siguiendo las preferencias de cada perfil.
    """
    course_by_cat = courses.groupby("category")["course_id"].apply(list).to_dict()
    records = []

    for _, user in users.iterrows():
        prefs = USER_PROFILES[user["profile"]]
        n_interactions = np.random.randint(5, 20)

        selected_courses: list[str] = []
        for _ in range(n_interactions):
            cat = np.random.choice(list(prefs.keys()), p=list(prefs.values()))
            if cat in course_by_cat:
                cid = random.choice(course_by_cat[cat])
                if cid not in selected_courses:
                    selected_courses.append(cid)

        for cid in selected_courses:
            course_row = courses[courses["course_id"] == cid].iloc[0]
            base_rating = course_row["avg_rating"]
            noise = np.random.normal(0, 0.5)
            rating = float(np.clip(round(base_rating + noise, 1), 1.0, 5.0))

            completed = rating >= 3.5
            progress = 100.0 if completed else round(random.uniform(10, 90), 1)
            sessions = np.random.randint(1, 12)
            session_min = round(random.uniform(10, 90), 1)

            records.append({
                "user_id":          user["user_id"],
                "course_id":        cid,
                "rating":           rating,
                "progress_pct":     progress,
                "sessions":         sessions,
                "session_duration": session_min,
                "completed":        int(completed),
            })

    return pd.DataFrame(records)


def save_to_sqlite(users: pd.DataFrame, courses: pd.DataFrame, interactions: pd.DataFrame, db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    users.to_sql("users", conn, if_exists="replace", index=False)
    courses.to_sql("courses", conn, if_exists="replace", index=False)
    interactions.to_sql("interactions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Base de datos guardada en: {db_path}")


def main():
    print("Generando datos sintéticos...")
    users = generate_users()
    courses = generate_courses()
    interactions = generate_interactions(users, courses)

    print(f"  Usuarios: {len(users)}")
    print(f"  Cursos:   {len(courses)}")
    print(f"  Interacciones: {len(interactions)}")

    base = os.path.dirname(__file__)
    db_path = os.path.join(base, "..", "database", "edtech.db")
    save_to_sqlite(users, courses, interactions, db_path)

    # También guardar CSV para referencia
    csv_dir = os.path.join(base)
    users.to_csv(os.path.join(csv_dir, "users.csv"), index=False)
    courses.to_csv(os.path.join(csv_dir, "courses.csv"), index=False)
    interactions.to_csv(os.path.join(csv_dir, "interactions.csv"), index=False)
    print("CSV exportados en data/")


if __name__ == "__main__":
    main()
