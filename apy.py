import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import ConfusionMatrixDisplay
# CONFIGURACION DEL APP
st.set_page_config(
    page_title="Deserción Estudiantil",
    layout="wide"
)

st.title(" Análisis Exploratorio y Predicción de Deserción Estudiantil")

## Carga del dataset
df = pd.read_excel(
    "C:/Users/salaz/OneDrive/Datos adjuntos/Proyecto MINERIA/REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx"
)

# Normalizar nombres de columnas (igual que en el entrenamiento)
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

st.success("Archivo cargado correctamente")

# EDA (VISTA GENERAL)
st.subheader(" Vista general del dataset")
st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
st.dataframe(df.head())

# EDA (COLUMNAS DEL DATASET)
st.subheader(" Columnas del dataset")
st.write(df.columns.tolist())

# EDA (INFORMACION GENERAL DEL DATASET)
st.subheader("ℹ Información general del dataset")
st.write("Filas:", df.shape[0])
st.write("Columnas:", df.shape[1])
st.write("Columnas del dataset:")
st.write(list(df.columns))

# EDA (VALORES NULOS POR COLUMNAS)
st.subheader(" Valores nulos por columna")
st.dataframe(df.isnull().sum())

#EDA (ESTADISTICAS DESCRIPTIVAS)
st.subheader(" Estadísticas descriptivas")
st.dataframe(df.describe())

# EDA (DISTRIBUCION DEL ESTADO)
st.subheader(" Estado del estudiante")

estado_counts = df["ESTADO"].value_counts()

fig, ax = plt.subplots(figsize=(3, 4))
ax.pie(
    estado_counts,
    labels=estado_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
ax.axis("equal")
st.pyplot(fig)

# EVALUAMOS EL MODELO 
st.subheader(" Evaluación del modelo predictivo")

modelo = joblib.load("modelRegresionLogistic.pkl")
st.success("Modelo cargado correctamente")

accuracy = 0.57
precision = 0.52
recall = 0.83
f1 = 0.64

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1-score", f"{f1:.2f}")

conf_matrix = np.array([[40, 75],
                        [17, 81]])

fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay(
    conf_matrix,
    display_labels=["No deserta", "Deserta"]
).plot(ax=ax_cm)
st.pyplot(fig_cm)

#Aqui vamos a predecir la Desercion 
st.subheader("🧑‍🎓 Predicción de riesgo de deserción")

st.write("""
Ingrese los datos académicos del estudiante para estimar
el riesgo de deserción según el modelo entrenado.
""")

# Valores válidos según el dataset
facultades = df["FACULTAD"].dropna().unique()
carreras = df["CARRERA"].dropna().unique()

with st.form("form_prediccion"):

    facultad = st.selectbox("Facultad", facultades)
    carrera = st.selectbox("Carrera", carreras)

    promedio = st.number_input(
        "Promedio académico",
        min_value=0.0,
        max_value=10.0,
        step=0.1
    )

    asistencia = st.number_input(
        "Asistencia (%)",
        min_value=0.0,
        max_value=100.0,
        step=1.0
    )

    nivel = st.number_input(
        "Nivel académico",
        min_value=1,
        step=1
    )

    submit = st.form_submit_button("Predecir riesgo")

if submit:
    datos_estudiante = pd.DataFrame([{
        "PROMEDIO": promedio,
        "ASISTENCIA": asistencia,
        "NIVEL": nivel,
        "FACULTAD": facultad,
        "CARRERA": carrera
    }])

    prediccion = modelo.predict(datos_estudiante)[0]
    probabilidad = modelo.predict_proba(datos_estudiante)[0][1]

    if probabilidad < 0.45:
        st.success(f"🟢 Riesgo BAJO ({probabilidad*100:.1f}%)")
    elif probabilidad < 0.65:
        st.warning(f"🟡 Riesgo MEDIO ({probabilidad*100:.1f}%)")
    else:
        st.error(f"🔴 Riesgo ALTO ({probabilidad*100:.1f}%)")

# VAMOS A LA PARTE DE LA IMPORTANCIA DE LAS VARIABLES
st.subheader(" Importancia de las variables")

coeficientes = modelo.named_steps["classifier"].coef_[0]
variables = modelo.named_steps["preprocess"].get_feature_names_out()

importancia_df = pd.DataFrame({
    "Variable": variables,
    "Importancia": coeficientes
}).sort_values(by="Importancia", ascending=False)

st.dataframe(importancia_df)

fig_imp, ax_imp = plt.subplots(figsize=(6, 5))
ax_imp.barh(importancia_df["Variable"], importancia_df["Importancia"])
ax_imp.invert_yaxis()
st.pyplot(fig_imp)