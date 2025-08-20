import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# ======================
# CONFIGURACIÓN GENERAL
# ======================
st.set_page_config(page_title="ML Interactivo: EDA + KNN/Tree/NB", layout="wide")
st.title("🧠 ML Interactivo con EDA (KNN, Árbol de Decisión, Naive Bayes)")

# ======================
# SIDEBAR – CONTROLES
# ======================
st.sidebar.header("⚙️ Configuración de Datos")
n_samples = st.sidebar.slider("Número de muestras", 300, 3000, 600, step=50)
n_features = st.sidebar.slider("Número de features totales", 6, 20, 8)
n_informative = st.sidebar.slider("Features informativas", 2, min(10, n_features), min(4, n_features))
n_redundant = st.sidebar.slider("Features redundantes", 0, max(0, n_features - n_informative), 0)
n_classes = st.sidebar.selectbox("Clases", [2, 3], index=0)
class_sep = st.sidebar.slider("Separación de clases", 0.5, 3.0, 1.0, 0.1)
flip_y = st.sidebar.slider("Ruido (etiquetas mal etiquetadas)", 0.0, 0.3, 0.01, 0.01)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42)

st.sidebar.header("🧪 Split & Escalado")
test_size = st.sidebar.slider("Proporción Test", 0.1, 0.5, 0.3, 0.05)
use_scaler = st.sidebar.checkbox("Estandarizar (recomendado para KNN y NB)", True)

st.sidebar.header("🧩 Visualización")
viz_mode = st.sidebar.selectbox("Espacio para visualización", ["PCA (2D)", "Elegir 2 features"])
feat_for_plot = (0, 1)
if viz_mode == "Elegir 2 features":
    c1 = st.sidebar.number_input("Feature X (índice)", 0, n_features - 1, 0)
    c2 = st.sidebar.number_input("Feature Y (índice)", 0, n_features - 1, 1)
    feat_for_plot = (int(c1), int(c2))

show_pairplot = st.sidebar.checkbox("Pairplot (muestras ↓ para velocidad)", False)
pairplot_sample = st.sidebar.slider("Muestras para pairplot", 100, min(1000, n_samples), min(400, n_samples), 50)

st.sidebar.header("🤖 Hiperparámetros")
st.sidebar.subheader("KNN")
knn_k = st.sidebar.slider("n_neighbors", 1, 31, 5, 2)
knn_weights = st.sidebar.selectbox("weights", ["uniform", "distance"])

st.sidebar.subheader("Árbol de Decisión")
tree_max_depth = st.sidebar.slider("max_depth", 1, 30, 5)
tree_criterion = st.sidebar.selectbox("criterion", ["gini", "entropy", "log_loss"])

st.sidebar.subheader("Naive Bayes")
# GaussianNB no tiene hiperparámetros críticos, pero habilitamos var_smoothing como ajuste fino
nb_smoothing = st.sidebar.number_input("var_smoothing", min_value=1e-12, max_value=1e-3, value=1e-9, format="%.1e")

# ======================
# GENERACIÓN/INGESTA DE DATOS
# ======================
st.header("📊 1) Generación de Datos (Simulados)")

# Asegurar límites válidos
n_redundant = min(n_redundant, max(0, n_features - n_informative))
n_repeated = 0
n_clusters_per_class = 2 if n_classes == 2 else 1

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_repeated=n_repeated,
    n_classes=n_classes,
    n_clusters_per_class=n_clusters_per_class,
    class_sep=class_sep,
    flip_y=flip_y,
    random_state=random_state,
)

columns = [f"feature_{i}" for i in range(n_features)]
df = pd.DataFrame(X, columns=columns)
df["target"] = y

c1, c2 = st.columns([2, 1])
with c1:
    st.write("Vista previa del dataset:")
    st.dataframe(df.head())
with c2:
    st.metric("Muestras", len(df))
    st.metric("Features", n_features)
    cls_counts = pd.Series(y).value_counts().sort_index()
    st.write("Distribución de clases:")
    st.bar_chart(cls_counts)

# Descarga del dataset
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar dataset (CSV)", data=csv, file_name="dataset_simulado.csv", mime="text/csv")

# ======================
# 2) EDA RÁPIDO
# ======================
st.header("🔎 2) EDA (Exploratory Data Analysis)")

eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["Resumen", "Distribuciones", "Correlación", "Pairplot (opcional)"])

with eda_tab1:
    st.subheader("Resumen estadístico")
    st.dataframe(df.describe().T)

    st.subheader("Valores faltantes por columna")
    st.dataframe(df.isna().sum())

with eda_tab2:
    st.subheader("Histogramas por feature")
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.ravel()
    for i, col in enumerate(columns):
        axes[i].hist(df[col], bins=30, alpha=0.8)
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    st.pyplot(fig)

with eda_tab3:
    st.subheader("Matriz de correlación")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[columns].corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with eda_tab4:
    st.subheader("Pairplot (submuestreo)")
    if show_pairplot:
        df_sample = df.sample(n=min(pairplot_sample, len(df)), random_state=random_state)
        fig = sns.pairplot(df_sample, vars=columns[:min(5, n_features)], hue="target", corner=True)
        st.pyplot(fig)

# ======================
# 3) TRAIN/TEST SPLIT
# ======================
X = df[columns].values
y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# ======================
# 4) MODELOS Y PIPELINES
# ======================
def make_pipeline(model):
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)

models = {
    "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights)),
    "Árbol de Decisión": make_pipeline(DecisionTreeClassifier(max_depth=tree_max_depth, criterion=tree_criterion, random_state=random_state)),
    "Naive Bayes": make_pipeline(GaussianNB(var_smoothing=nb_smoothing)),
}

# Entrenamiento + evaluación
st.header("🤖 3) Entrenamiento y Evaluación")
results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"pipeline": pipe, "acc": acc, "y_pred": y_pred}

# Comparativa de accuracy
acc_df = pd.DataFrame({"Modelo": list(results.keys()), "Accuracy": [v["acc"] for v in results.values()]})
st.subheader("Comparativa de Accuracy")
st.bar_chart(acc_df.set_index("Modelo"))

# Reportes y matrices
st.subheader("Reportes y Matrices de Confusión")
for name, res in results.items():
    st.markdown(f"**{name}** — Accuracy: `{res['acc']:.3f}`")
    report = classification_report(y_test, res["y_pred"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T)

    cm = confusion_matrix(y_test, res["y_pred"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {name}")
    st.pyplot(fig)
    st.markdown("---")

# ======================
# 5) VISUALIZACIONES 2D: DISPERSIÓN + FRONTERAS
# ======================
st.header("🗺️ 4) Visualizaciones 2D: Dispersión y Fronteras de Decisión")

# Preparar espacio 2D
if viz_mode == "PCA (2D)":
    pca = PCA(n_components=2, random_state=random_state)
    X2d = pca.fit_transform(X)
    x_label, y_label = "PCA 1", "PCA 2"
else:
    X2d = X[:, [feat_for_plot[0], feat_for_plot[1]]]
    x_label, y_label = columns[feat_for_plot[0]], columns[feat_for_plot[1]]

# Dispersión completa
st.subheader("Dispersión del dataset en 2D")
fig, ax = plt.subplots()
scatter = ax.scatter(X2d[:, 0], X2d[:, 1], c=y, edgecolor="k", alpha=0.7)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_title("Dispersión (todas las muestras)")
st.pyplot(fig)

# Fronteras por modelo (reentrenamos en 2D SOLO para graficar)
xx_min, xx_max = X2d[:, 0].min() - 1, X2d[:, 0].max() + 1
yy_min, yy_max = X2d[:, 1].min() - 1, X2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, 300), np.linspace(yy_min, yy_max, 300))

for name in models.keys():
    # Construir pipeline coherente con el escalado
    if name == "KNN":
        clf2d = make_pipeline(KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights))
    elif name == "Árbol de Decisión":
        clf2d = make_pipeline(DecisionTreeClassifier(max_depth=tree_max_depth, criterion=tree_criterion, random_state=random_state))
    else:
        clf2d = make_pipeline(GaussianNB(var_smoothing=nb_smoothing))

    X2d_train, X2d_test, y2d_train, y2d_test = train_test_split(X2d, y, test_size=test_size, random_state=random_state, stratify=y)
    clf2d.fit(X2d_train, y2d_train)
    Z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X2d_test[:, 0], X2d_test[:, 1], c=y2d_test, edgecolor="k", alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Fronteras de decisión — {name} (espacio 2D)")
    st.pyplot(fig)

# ======================
# 6) ÁRBOL DE DECISIÓN: ESTRUCTURA
# ======================
st.header("🌳 5) Visualización del Árbol de Decisión")
# Para ver atributos originales, usamos el modelo del pipeline entrenado en el espacio completo
tree_model = results["Árbol de Decisión"]["pipeline"].named_steps["model"]
fig, ax = plt.subplots(figsize=(18, 10))
plot_tree(
    tree_model,
    feature_names=columns,
    class_names=[str(c) for c in sorted(np.unique(y))],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)
st.pyplot(fig)

st.caption("Tip: usa un `max_depth` moderado para árboles legibles.")

# ======================
# FIN
# ======================
st.success("Listo ✅ Juega con los controles en la izquierda para personalizar datos y modelos.")
