# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# Título
st.title("🌳 Clasificación con Árbol de Decisión")
st.write("Sube tu archivo CSV para entrenar un modelo de Árbol de Decisión.")

# Subida de archivo
uploaded_file = st.file_uploader("📂 Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Vista previa de los datos")
    st.write(df.head())

    # Selección de columna objetivo
    target_column = st.selectbox("Selecciona la columna objetivo (clase a predecir):", df.columns)

    # Variables predictoras y objetivo
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Parámetros del Árbol
    st.sidebar.header("⚙️ Parámetros del Árbol")
    criterion = st.sidebar.selectbox("Criterio de división:", ["gini", "entropy", "log_loss"])
    max_depth = st.sidebar.slider("Profundidad máxima del árbol:", 1, 20, 3)
    test_size = st.sidebar.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.3)

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Entrenamiento
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Predicciones
    y_pred = clf.predict(X_test)

    # Resultados
    st.subheader("📈 Resultados del Modelo")
    st.write(f"**Exactitud (Accuracy):** {accuracy_score(y_test, y_pred):.2f}")
    st.text("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))

    # Visualización del Árbol
    st.subheader("🌳 Visualización del Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(c) for c in clf.classes_], ax=ax)
    st.pyplot(fig)

else:
    st.info("Por favor, sube un archivo CSV para continuar.")
