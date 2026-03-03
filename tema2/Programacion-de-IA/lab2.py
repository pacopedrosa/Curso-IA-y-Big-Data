import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from google.colab import files

uploaded = files.upload()

# cargar dataset
df = pd.read_csv('abandono_escolar.csv')
print(df.head())

#verificar valores nulos
print(df.isnull().sum())

#ver distribucion de la variable objetivo
sns.countplot(x='abandono', data=df)
plt.title('Distribucion de estudiantes que abandonan o no')
plt.show()

#Variables predictoras y objetivo
x = df.drop(columns=['abandono'])
y = df['abandono']

#Normalizar datos
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print(f'Precisión del modelo: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Matriz de confusión
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()