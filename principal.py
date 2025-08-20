import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# ======================
# Configuración de la página
# ======================
st.set_page_config(page_title="Clasificadores ML", layout="wide")
st.title("🧠 Clasificadores en Python (KNN, Árbol de Decisión, Naive Bayes)")

# ======================
# Crear dataset simulado
# ======================
st.header("📊 Generación de Datos Simulados")

X, y = make_classification(
    n_samples=300,     # mínimo 300 muestras
    n_features=6,      # mínimo 6 columnas
    n_informative=4,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 7)])
df["target"] = y

st.write("Vista previa del dataset:")
st.dataframe(df.head())

# ======================
# Dividir datos
# ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======================
# Reducir a 2D con PCA para visualización
# ======================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca = train_test_split(X_pca, test_size=0.3, random_state=42)

# ======================
# Modelos
# ======================
st.header("🤖 Modelos Entrenados")

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "model": model,
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "matrix": confusion_matrix(y_test, y_pred)
    }

# ======================
# Mostrar resultados
# ======================
for name, res in results.items():
    st.subheader(f"🔹 {name}")
    st.write(f"**Accuracy:** {res['accuracy']:.2f}")
    
    # Mostrar matriz de confusión como heatmap
    fig, ax = plt.subplots()
    sns.heatmap(res["matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión - {name}")
    st.pyplot(fig)

    # Mostrar clasificación detallada
    st.write("Reporte de Clasificación:")
    st.dataframe(pd.DataFrame(res["report"]).transpose())
    
    # ======================
    # Gráfica de dispersión PCA
    # ======================
    st.write("📍 Dispersión de datos (PCA)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Dispersión de datos - {name}")
    st.pyplot(fig)

    # ======================
    # Fronteras de decisión
    # ======================
    st.write("📉 Fronteras de decisión (PCA)")
    model = res["model"]
    model.fit(X_pca, y)  # entrenamos sobre datos reducidos a 2D para graficar fronteras

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Fronteras de decisión - {name}")
    st.pyplot(fig)

    st.markdown("---")
