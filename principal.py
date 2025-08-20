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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from io import StringIO
import graphviz

st.set_page_config(page_title="ML Models App", layout="wide")

st.title("🔬 Plataforma Interactiva de Modelos de Machine Learning")
st.markdown("### Especializada en clasificación con datos cualitativos")

st.sidebar.header("⚙️ Configuración")

# Opción de carga de CSV o dataset simulado
option = st.sidebar.radio("¿Cómo quieres cargar los datos?",
                          ("Usar dataset simulado", "Cargar un archivo CSV propio"))

# Variables globales para el preprocesamiento
encoders = {}
df_processed = None

if option == "Cargar un archivo CSV propio":
    st.sidebar.subheader("📂 Instrucciones para el CSV")
    st.sidebar.markdown("""
    Tu archivo **CSV** debe tener:
    - Al menos **2 columnas** (1 característica y 1 objetivo).
    - La **última columna** debe ser la variable objetivo (clase).
    - Puede contener datos cualitativos (texto) y cuantitativos (números).
    - Sin valores nulos.
    """)
    uploaded_file = st.file_uploader("Carga tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Vista previa de tus datos originales")
            st.dataframe(df.head())
            
            # Mostrar información del dataset
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Información del dataset:**")
                st.write(f"- Filas: {len(df)}")
                st.write(f"- Columnas: {len(df.columns)}")
                
            with col2:
                st.write("**Tipos de datos:**")
                for col in df.columns:
                    dtype = "Cualitativo" if df[col].dtype == 'object' else "Cuantitativo"
                    st.write(f"- {col}: {dtype}")
            
            # Verificar valores nulos
            if df.isnull().sum().sum() > 0:
                st.warning("⚠️ Tu dataset contiene valores nulos. Por favor, límpialos antes de continuar.")
                st.write("Valores nulos por columna:")
                st.write(df.isnull().sum())
                st.stop()
                
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            st.stop()
    else:
        st.warning("Por favor carga un archivo CSV válido.")
        st.stop()
else:
    # Crear dataset simulado con datos cualitativos - Evaluación de Empleados
    np.random.seed(42)
    n_samples = 300
    
    # Crear características cualitativas simuladas
    departamento = np.random.choice(['Ventas', 'Marketing', 'IT', 'RRHH', 'Finanzas'], n_samples)
    nivel_educacion = np.random.choice(['Secundaria', 'Universitario', 'Posgrado', 'Doctorado'], n_samples)
    experiencia = np.random.choice(['Junior', 'Semi-Senior', 'Senior'], n_samples)
    modalidad_trabajo = np.random.choice(['Presencial', 'Remoto', 'Híbrido'], n_samples)
    capacitacion = np.random.choice(['Baja', 'Media', 'Alta'], n_samples)
    
    # Crear algunas características cuantitativas también
    horas_trabajadas = np.random.normal(42, 8, n_samples)  # promedio 42 horas por semana
    proyectos_completados = np.random.poisson(12, n_samples)  # promedio 12 proyectos por año
    
    # Crear variable objetivo basada en las características con lógica empresarial
    evaluacion = []
    for i in range(n_samples):
        score = 0
        
        # Puntuación basada en departamento
        if departamento[i] == 'IT':
            score += 2
        elif departamento[i] in ['Ventas', 'Marketing']:
            score += 1
            
        # Puntuación basada en educación
        if nivel_educacion[i] == 'Doctorado':
            score += 3
        elif nivel_educacion[i] == 'Posgrado':
            score += 2
        elif nivel_educacion[i] == 'Universitario':
            score += 1
            
        # Puntuación basada en experiencia
        if experiencia[i] == 'Senior':
            score += 2
        elif experiencia[i] == 'Semi-Senior':
            score += 1
            
        # Puntuación basada en capacitación
        if capacitacion[i] == 'Alta':
            score += 2
        elif capacitacion[i] == 'Media':
            score += 1
            
        # Agregar algo de aleatoriedad
        score += np.random.randint(-2, 3)
        
        # Determinar evaluación final
        if score >= 6:
            evaluacion.append('Excelente')
        elif score >= 3:
            evaluacion.append('Bueno')
        else:
            evaluacion.append('Regular')
    
    # Asegurar que las horas trabajadas y proyectos sean positivos
    horas_trabajadas = np.clip(horas_trabajadas, 20, 60)
    proyectos_completados = np.clip(proyectos_completados, 1, 25)
    
    df = pd.DataFrame({
        'Departamento': departamento,
        'Nivel_Educacion': nivel_educacion,
        'Experiencia': experiencia,
        'Modalidad_Trabajo': modalidad_trabajo,
        'Capacitacion': capacitacion,
        'Horas_Semanales': horas_trabajadas.round(1),
        'Proyectos_Anuales': proyectos_completados,
        'Evaluacion': evaluacion
    })
    
    st.write("### Dataset simulado generado (Evaluación de Empleados)")
    st.dataframe(df.head())

# Preprocesamiento de datos
st.write("### 🔄 Preprocesamiento de datos")

# Separar características y objetivo
X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()

# Mostrar las columnas antes del preprocesamiento
st.write("**Características (X):**", list(X.columns))
st.write("**Variable objetivo (y):**", y.name if hasattr(y, 'name') else "target")

# Identificar columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

st.write(f"**Columnas categóricas:** {categorical_cols}")
st.write(f"**Columnas numéricas:** {numerical_cols}")

# Aplicar codificación a variables categóricas
X_processed = X.copy()

if categorical_cols:
    st.write("#### Codificación de variables categóricas")
    encoding_method = st.selectbox(
        "Selecciona el método de codificación:",
        ["Label Encoding", "Ordinal Encoding (personalizado)"]
    )
    
    if encoding_method == "Label Encoding":
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col])
            encoders[col] = le
            st.write(f"- {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    else:  # Ordinal Encoding personalizado
        st.write("Define el orden para cada variable categórica:")
        for col in categorical_cols:
            unique_vals = sorted(X[col].unique())
            st.write(f"**{col}:** {unique_vals}")
            # Para simplicidad, usar orden alfabético por defecto
            oe = OrdinalEncoder(categories=[unique_vals])
            X_processed[[col]] = oe.fit_transform(X[[col]])
            encoders[col] = oe

# Codificar variable objetivo si es categórica
if y.dtype == 'object':
    le_target = LabelEncoder()
    y_processed = le_target.fit_transform(y)
    encoders['target'] = le_target
    st.write(f"**Variable objetivo codificada:** {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
else:
    y_processed = y.copy()

# Mostrar datos procesados
st.write("### Vista previa de datos procesados")
df_processed = X_processed.copy()
df_processed['target'] = y_processed
st.dataframe(df_processed.head())

# Train-Test Split
test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%)", 10, 50, 30, step=5)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, test_size=test_size/100, random_state=42, stratify=y_processed
)

# Selección de modelo
st.sidebar.subheader("🤖 Selección de modelo")
model_choice = st.sidebar.selectbox("Elige un modelo",
                                    ("Árbol de Decisión", "KNN", "Naive Bayes"))

if model_choice == "Árbol de Decisión":
    st.sidebar.markdown("### Parámetros del Árbol de Decisión")
    criterion = st.sidebar.selectbox("Criterio", ("gini", "entropy", "log_loss"))
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 15, 5)
    min_samples_split = st.sidebar.slider("Mínimo muestras para dividir", 2, 20, 2)
    min_samples_leaf = st.sidebar.slider("Mínimo muestras por hoja", 1, 10, 1)
    model = DecisionTreeClassifier(
        criterion=criterion, 
        max_depth=max_depth,
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
elif model_choice == "KNN":
    k = st.sidebar.slider("Número de vecinos (k)", 1, min(15, len(X_train)), 5)
    model = KNeighborsClassifier(n_neighbors=k)
else:
    model = GaussianNB()

# Entrenar modelo
with st.spinner('Entrenando modelo...'):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Resultados
st.subheader("📊 Resultados del Modelo")

col1, col2 = st.columns(2)

with col1:
    st.write("**Precisión del modelo:**")
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy:.3f}")
    
    st.write("**Matriz de confusión:**")
    cm = confusion_matrix(y_test, y_pred)
    
    # Crear heatmap de la matriz de confusión
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    
    # Obtener las etiquetas originales si existen
    if 'target' in encoders:
        labels = encoders['target'].classes_
    else:
        labels = sorted(np.unique(y_processed))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel('Predicción')
    ax_cm.set_ylabel('Valor Real')
    ax_cm.set_title('Matriz de Confusión')
    st.pyplot(fig_cm)

with col2:
    st.write("**Reporte de clasificación:**")
    if 'target' in encoders:
        target_names = encoders['target'].classes_
        report = classification_report(y_test, y_pred, target_names=target_names)
    else:
        report = classification_report(y_test, y_pred)
    st.text(report)

# Importancia de características (solo para árboles de decisión)
if model_choice == "Árbol de Decisión":
    st.subheader("📈 Importancia de Características")
    feature_importance = pd.DataFrame({
        'Característica': X_processed.columns,
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importancia', y='Característica', ax=ax_imp)
    ax_imp.set_title('Importancia de Características en el Árbol de Decisión')
    st.pyplot(fig_imp)

# Visualización de dispersión (solo para columnas numéricas)
if len(numerical_cols) >= 2:
    st.subheader("🌐 Gráfica de Dispersión")
    feature_x = st.selectbox("Selecciona eje X", numerical_cols)
    feature_y = st.selectbox("Selecciona eje Y", [col for col in numerical_cols if col != feature_x])

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Usar etiquetas originales para el color si existen
    if 'target' in encoders:
        hue_data = [encoders['target'].classes_[i] for i in y_processed]
        legend_labels = encoders['target'].classes_
    else:
        hue_data = y_processed
        legend_labels = None
    
    scatter = ax.scatter(df[feature_x], df[feature_y], c=y_processed, cmap='viridis', alpha=0.7)
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f'Dispersión: {feature_x} vs {feature_y}')
    
    # Añadir colorbar
    if legend_labels is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=plt.cm.viridis(i/len(legend_labels)), 
                             markersize=10) for i in range(len(legend_labels))]
        ax.legend(handles, legend_labels, title='Clase')
    else:
        plt.colorbar(scatter, ax=ax)
    
    st.pyplot(fig)

# Mostrar árbol de decisión
if model_choice == "Árbol de Decisión":
    st.subheader("🌳 Visualización del Árbol de Decisión")
    
    try:
        # Preparar nombres de características y clases
        feature_names = list(X_processed.columns)
        if 'target' in encoders:
            class_names = [str(c) for c in encoders['target'].classes_]
        else:
            class_names = [str(c) for c in sorted(np.unique(y_processed))]
        
        dot_data = export_graphviz(
            model, 
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True, 
            rounded=True,
            special_characters=True,
            max_depth=3  # Limitar profundidad para mejor visualización
        )
        
        st.graphviz_chart(dot_data)
        
        # Opción para descargar el árbol
        st.download_button(
            label="📥 Descargar visualización del árbol (DOT)",
            data=dot_data,
            file_name="decision_tree.dot",
            mime="text/plain"
        )
        
    except Exception as e:
        st.error(f"Error al generar la visualización del árbol: {str(e)}")

# Sección de predicción interactiva
st.subheader("🎯 Hacer Predicciones")
st.write("Ingresa valores para hacer una predicción:")

prediction_data = {}
cols = st.columns(min(3, len(X.columns)))

for i, col in enumerate(X.columns):
    with cols[i % 3]:
        if col in categorical_cols:
            unique_vals = sorted(X[col].unique())
            prediction_data[col] = st.selectbox(f"Selecciona {col}", unique_vals)
        else:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            prediction_data[col] = st.number_input(
                f"Ingresa {col}", 
                min_value=min_val, 
                max_value=max_val, 
                value=mean_val
            )

if st.button("🔮 Hacer Predicción"):
    # Preparar datos para predicción
    pred_df = pd.DataFrame([prediction_data])
    
    # Aplicar el mismo preprocesamiento
    pred_processed = pred_df.copy()
    for col in categorical_cols:
        if col in encoders:
            if encoding_method == "Label Encoding":
                pred_processed[col] = encoders[col].transform(pred_df[col])
            else:
                pred_processed[[col]] = encoders[col].transform(pred_df[[col]])
    
    # Hacer predicción
    prediction = model.predict(pred_processed)
    prediction_proba = model.predict_proba(pred_processed) if hasattr(model, 'predict_proba') else None
    
    # Mostrar resultado
    if 'target' in encoders:
        predicted_class = encoders['target'].inverse_transform(prediction)[0]
    else:
        predicted_class = prediction[0]
    
    st.success(f"🎯 Predicción: **{predicted_class}**")
    
    if prediction_proba is not None:
        st.write("**Probabilidades por clase:**")
        if 'target' in encoders:
            classes = encoders['target'].classes_
        else:
            classes = sorted(np.unique(y_processed))
        
        prob_df = pd.DataFrame({
            'Clase': classes,
            'Probabilidad': prediction_proba[0]
        }).sort_values('Probabilidad', ascending=False)
        
        st.dataframe(prob_df)

# Información adicional
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.markdown("""
Esta aplicación está optimizada para trabajar con datos cualitativos usando:
- **Árboles de Decisión**: Ideales para datos categóricos
- **Codificación automática** de variables categóricas
- **Visualización interactiva** del árbol
- **Predicciones en tiempo real**
""")

if st.sidebar.button("🔄 Reiniciar aplicación"):
    st.experimental_rerun()
