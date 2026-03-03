import pandas as pd
import numpy as np

#fijamos la semilla para la reproducibilidad
np.random.seed(42)

#generamos 1000 registros
num_alumnos = 1000
data = {
    "calificacion_promedio": np.round(np.random.uniform(3, 10, num_alumnos), 2),
    "asistencia": np.round(np.random.uniform(50, 100, num_alumnos), 2),
    "horas_estudio": np.random.randint(0, 6, num_alumnos),
    "nivel_socioeconomico": np.random.choice([1, 2, 3], num_alumnos),
    "actividades_extracurriculares": np.random.choice([0, 1], num_alumnos),
    "problemas_conducta": np.random.choice([0, 1], num_alumnos)
}
df = pd.DataFrame(data)

# Reglas para determinar abandono escolar (de forma simulada)
df["abandono"] = ((df["calificacion_promedio"] < 5) & (df["asistencia"] < 70)) | (df["problemas_conducta"] == 1)
df["abandono"] = df["abandono"].astype(int)  # Convertimos a valores 0 y 1

# Guardamos en un archivo CSV
df.to_csv("abandono_escolar.csv", index=False)
