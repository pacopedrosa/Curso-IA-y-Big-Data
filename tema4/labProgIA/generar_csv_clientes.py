import pandas as pd
import numpy as np

# Semilla para reproducibilidad
np.random.seed(42)

n_clientes = 200

datos = {
    'id_cliente': np.arange(1, n_clientes+1),
    'edad': np.random.randint(18, 70, n_clientes),
    'genero': np.random.choice(['M', 'F'], n_clientes),
    'ingresos_anuales': np.random.normal(30000, 12000, n_clientes).astype(int),
    'frecuencia_compra': np.random.randint(1, 30, n_clientes),
    'boleto_medio': np.random.normal(60, 25, n_clientes).round(2)
}

df = pd.DataFrame(datos)
# Limitar ingresos y boleto medio a valores positivos
for col in ['ingresos_anuales', 'boleto_medio']:
    df[col] = df[col].apply(lambda x: abs(x))

# Guardar como CSV
csv_path = 'clientes.csv'
df.to_csv(csv_path, index=False)
print(f"Archivo CSV generado: {csv_path}")
