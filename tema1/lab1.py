import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Carga de datos original
data = pd.read_csv("reviews.csv")
print(data.head())

# Dividir datos ANTES de añadir nuevas reviews
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Añadir nuevas reviews SOLO al entrenamiento
nuevas_reviews = ["La calidad es fantastica", "Muy mal servicio, no lo recomiendo"]
nuevos_sentimientos = ["positive", "negative"]
nuevos_datos = pd.DataFrame({'review': nuevas_reviews, 'sentiment': nuevos_sentimientos})
train_data_ampliado = pd.concat([train_data, nuevos_datos], ignore_index=True)

# Vectorizar usando SOLO el vocabulario del entrenamiento ampliado
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train_data_ampliado['review'])
y_train = train_data_ampliado['sentiment']
x_test = vectorizer.transform(test_data['review'])
y_test = test_data['sentiment']

# Entrenar y evaluar antes de añadir nuevas reviews
vectorizer_base = CountVectorizer()
x_train_base = vectorizer_base.fit_transform(train_data['review'])
y_train_base = train_data['sentiment']
x_test_base = vectorizer_base.transform(test_data['review'])
y_test_base = test_data['sentiment']

model_base = MultinomialNB()
model_base.fit(x_train_base, y_train_base)
accuracy_before = model_base.score(x_test_base, y_test_base)
print(f'Precisión del modelo antes de añadir nuevas reviews: {accuracy_before * 100:.2f}%')

# Entrenar y evaluar después de añadir nuevas reviews
model = MultinomialNB()
model.fit(x_train, y_train)
accuracy_after = model.score(x_test, y_test)
print(f'Precisión del modelo después de añadir nuevas reviews: {accuracy_after * 100:.2f}%')

# Realizado por mi para hacer prueba e ir jugando
# Pedir al usuario una review y predecir su sentimiento
nueva_review_usuario = input("Escribe una nueva review para analizar su sentimiento: ")
nueva_review_vectorizada = vectorizer.transform([nueva_review_usuario])
prediccion = model.predict(nueva_review_vectorizada)[0]
print(f"El sentimiento de la review es: {prediccion}")