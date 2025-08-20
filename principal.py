import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# ======================
# Función para cargar datos
# ======================
@st.cache_data
def load_default_data():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4,
        n_redundant=0, n_classes=3, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    df["Target"] = y
    return df

# ======================
# App principal
# ======================
st.title("📊 Clasificación con Machine Learning")
st.write("KNN | Árbol de Decisión | Naive Bayes")

# Opción de carga de datos
uploaded_file = st.file_uploader("📂 Carga tu propio CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV cargado con éxito")
else:
    df = load_default_data()
    st.info("⚠️ No subiste un archivo. Se está usando un dataset simulado.")

# Mostrar preview
st.subheader("🔎 Vista previa de los datos")
st.dataframe(df.head())

# ======================
# EDA
# ======================
st.subheader("📈 Análisis Exploratorio de Datos (EDA)")

if st.checkbox("Mostrar descripción estadística"):
    st.write(df.describe())

if st.checkbox("Mostrar correlación entre variables"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if st.checkbox("Mostrar histogramas"):
    df.hist(figsize=(10, 6))
    st.pyplot(plt.gcf())

if st.checkbox("Mostrar pairplot"):
    st.write("⚠️ Esto puede tardar con muchos datos")
    fig = sns.pairplot(df, hue="Target")
    st.pyplot(fig)

# ======================
# Selección de variables
# ======================
features = st.multiselect("Selecciona las variables predictoras", df.columns[:-1], default=df.columns[:-1].tolist())
target = st.selectbox("Selecciona la variable objetivo", df.columns, index=len(df.columns)-1)

X = df[features]
y = df[target]

# ======================
# División de datos
# ======================
test_size = st.slider("Proporción de test", 0.1, 0.5, 0.3)
random_state = st.number_input("Random state", value=42, step=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# ======================
# Selección de modelo
# ======================
st.subheader("⚙️ Modelado")
model_name = st.selectbox("Elige un modelo", ["KNN", "Árbol de decisión", "Naive Bayes"])

if model_name == "KNN":
    n_neighbors = st.slider("Número de vecinos (K)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

elif model_name == "Árbol de decisión":
    max_depth = st.slider("Profundidad máxima", 1, 10, 3)
    criterion = st.selectbox("Criterio", ["gini", "entropy"])
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)

else:  # Naive Bayes
    model = GaussianNB()

# Entrenar modelo
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ======================
# Resultados
# ======================
st.subheader("📊 Resultados del modelo")
st.text(classification_report(y_test, y_pred))

# Matriz de confusión
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
st.pyplot(fig)

# ======================
# Visualización de fronteras de decisión
# ======================
if len(features) == 2:
    st.subheader("🌐 Fronteras de decisión")
    x_min, x_max = X[features[0]].min() - 1, X[features[0]].max() + 1
    y_min, y_max = X[features[1]].min() - 1, X[features[1]].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X_test[features[0]], X_test[features[1]], c=y_test, edgecolor="k", cmap=plt.cm.coolwarm)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    st.pyplot(fig)
else:
    st.info("⚠️ Para ver fronteras de decisión selecciona exactamente 2 variables predictoras.")

# ======================
# Visualización del árbol
# ======================
if model_name == "Árbol de decisión":
    st.subheader("🌳 Visualización del Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, filled=True, feature_names=features, class_names=[str(c) for c in np.unique(y)], ax=ax)
    st.pyplot(fig)
