import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from io import StringIO
import pydotplus
from sklearn import tree
import graphviz

st.set_page_config(page_title="ML Models App", layout="wide")

st.title("🔬 Plataforma Interactiva de Modelos de Machine Learning")

st.sidebar.header("⚙️ Configuración")

# Opción de carga de CSV o dataset simulado
option = st.sidebar.radio("¿Cómo quieres cargar los datos?",
                          ("Usar dataset simulado", "Cargar un archivo CSV propio"))

if option == "Cargar un archivo CSV propio":
    st.sidebar.subheader("📂 Instrucciones para el CSV")
    st.sidebar.markdown("""
    Tu archivo **CSV** debe tener:
    - Al menos **6 columnas** (5 características y 1 objetivo).
    - La **última columna** debe ser la variable objetivo (clase).
    - Sin valores nulos.
    """)
    uploaded_file = st.file_uploader("Carga tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Vista previa de tus datos")
        st.dataframe(df.head())
    else:
        st.warning("Por favor carga un archivo CSV válido.")
        st.stop()
else:
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                               n_redundant=0, n_classes=3, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)])
    df["target"] = y
    st.write("### Dataset simulado generado")
    st.dataframe(df.head())

# Separar X e y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-Test Split
test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%)", 10, 50, 30, step=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Selección de modelo
st.sidebar.subheader("🤖 Selección de modelo")
model_choice = st.sidebar.selectbox("Elige un modelo",
                                    ("KNN", "Naive Bayes", "Árbol de Decisión"))

if model_choice == "KNN":
    k = st.sidebar.slider("Número de vecinos (k)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)
elif model_choice == "Naive Bayes":
    model = GaussianNB()
else:
    st.sidebar.markdown("### Parámetros del Árbol de Decisión")
    criterion = st.sidebar.selectbox("Criterio", ("gini", "entropy"))
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 10, 3)
    min_samples_split = st.sidebar.slider("Mínimo muestras para dividir", 2, 10, 2)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                   min_samples_split=min_samples_split, random_state=42)

# Entrenar modelo
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Resultados
st.subheader("📊 Resultados")
st.text("Matriz de confusión:")
st.write(confusion_matrix(y_test, y_pred))
st.text("Reporte de clasificación:")
st.text(classification_report(y_test, y_pred))

# Visualización de dispersión
st.subheader("🌐 Gráficas")
feature_x = st.selectbox("Selecciona eje X", X.columns)
feature_y = st.selectbox("Selecciona eje Y", X.columns)

fig, ax = plt.subplots()
sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df.iloc[:, -1], palette="viridis", ax=ax)
st.pyplot(fig)

# Mostrar árbol de decisión
if model_choice == "Árbol de Decisión":
    st.subheader("🌳 Visualización del Árbol de Decisión")
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=X.columns,
                               class_names=[str(c) for c in np.unique(y)],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(dot_data)
