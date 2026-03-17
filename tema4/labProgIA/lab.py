import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Cargar datos
df = pd.read_csv('clientes.csv')

# EDA basico
print('Primeras filas del dataset:')
print(df.head())
print('\nInformación del dataset:')
print(df.info())
print('\nEstadísticas descriptivas:')
print(df.describe())
print('\nValores nulos por columna:')
print(df.isnull().sum())

X = df[['frecuencia_compra', 'boleto_medio']]

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])

# Método del codo para determinar el número óptimo de clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Gráfico del método del codo
plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.show()


# Elegimos k=3 basado en el gráfico del codo
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)


# Gráfico de dispersión de los clusters
plt.figure()
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=df['cluster'], cmap='viridis')
plt.xlabel('Frecuencia de compra (escalada)')
plt.ylabel('Boleto medio (escalado)')
plt.title('Clusters de clientes')
plt.show()

# Visualización de clusters
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=df['cluster'], cmap='tab10')
plt.xlabel('Frecuencia de compra (escalada)')
plt.ylabel('Boleto medio (escalado)')
plt.title('Clusters de clientes')
plt.savefig('clusters_clientes.png')
plt.show()

# Cuantos clientes hay en cada cluster y el promedio de frecuencia de compra y boleto medio por cluster
print(df['cluster'].value_counts())

# Promedio de frecuencia de compra y boleto medio por cluster
print(df.groupby('cluster')[['frecuencia_compra', 'boleto_medio']].mean())

print("\nInterpretación de los clusters y acciones de marketing sugeridas:\n")

print("Cluster 0: Clientes que compran con frecuencia media pero gastan mucho por compra.")
print("Acción: Ofertas VIP, programa de fidelidad, acceso anticipado a productos premium.\n")

print("Cluster 1: Clientes que compran muy frecuentemente pero gastan menos por compra.")
print("Acción: Promociones por volumen, cupones de descuento por compras frecuentes, packs o combos.\n")

print("Cluster 2: Clientes poco frecuentes y de bajo gasto.")
print("Acción: Campañas de reactivación, descuentos de bienvenida")