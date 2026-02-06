import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import ConfusionMatrixDisplay

#CONFIGURACION   
st.set_page_config(
    page_title="Deserción Estudiantil",
    layout="wide"
)

st.title("Dashboard de Deserción Estudiantil")
st.caption("Análisis exploratorio de datos y predicción de riesgo académico")

st.divider()

# CARGA DEL DATASET
df = pd.read_excel(
    "C:/Users/salaz/OneDrive/Datos adjuntos/Proyecto MINERIA/REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx"
)

# Normalización (igual que en el modelo)
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

# Limpieza de ESTADO
df["ESTADO"] = (
    df["ESTADO"]
    .astype(str)
    .str.strip()
    .str.upper()
)
df = df[df["ESTADO"].isin(["APROBADA", "REPROBADA"])]

st.success("Dataset cargado correctamente")

# RESUMEN GENERAL DEL DATASET
st.subheader("Resumen general del dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Total de registros", df.shape[0])
col2.metric("Total de columnas", df.shape[1])
col3.metric("Carreras distintas", df["CARRERA"].nunique())

st.divider()

# VISTA DE  LOS DATOS
st.subheader("Vista previa de los datos")
st.dataframe(df.head(), height=250)

st.divider()

# VALORES NULOS
st.subheader("Valores nulos por columna")
st.dataframe(df.isnull().sum())

st.divider()

# DISTRIBUCION DEL ESTADO ACADEMICO
st.subheader("Distribución del estado académico")

estado_counts = df["ESTADO"].value_counts()

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(
    estado_counts,
    labels=estado_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
ax.set_title("Estado académico de los estudiantes")
ax.axis("equal")
st.pyplot(fig)

st.divider()

# EVALUACION DEL MODELO
st.subheader("Evaluación del modelo predictivo")

modelo = joblib.load("modelRegresionLogistic.pkl")
st.success("Modelo cargado correctamente")

# Métricas obtenidas en entrenamiento
accuracy = 0.57
precision = 0.52
recall = 0.83
f1 = 0.64

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1-score", f"{f1:.2f}")

st.caption(
    "El modelo prioriza detectar estudiantes en riesgo (alto recall), "
    "aunque sacrifica precisión global."
)

# Matriz de confusión
conf_matrix = np.array([[40, 75],
                        [17, 81]])

fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay(
    conf_matrix,
    display_labels=["No deserta", "Deserta"]
).plot(ax=ax_cm)
st.pyplot(fig_cm)

st.divider()

# PREDICCION DEL RIESGO DE DESERCION
st.subheader(" Predicción de riesgo de deserción")

st.write(
    "Ingrese los datos académicos del estudiante para estimar "
    "el riesgo de deserción."
)

facultades = df["FACULTAD"].dropna().unique()
carreras = df["CARRERA"].dropna().unique()

with st.form("form_prediccion"):

    col1, col2 = st.columns(2)
    facultad = col1.selectbox("Facultad", facultades)
    carrera = col2.selectbox("Carrera", carreras)

    col3, col4, col5 = st.columns(3)
    promedio = col3.number_input("Promedio académico", 0.0, 10.0, step=0.1)
    asistencia = col4.number_input("Asistencia (%)", 0.0, 100.0, step=1.0)
    nivel = col5.number_input("Nivel académico", min_value=1, step=1)

    submit = st.form_submit_button("Predecir riesgo")

if submit:
    datos_estudiante = pd.DataFrame([{
        "PROMEDIO": promedio,
        "ASISTENCIA": asistencia,
        "NIVEL": nivel,
        "FACULTAD": facultad,
        "CARRERA": carrera
    }])

    probabilidad = modelo.predict_proba(datos_estudiante)[0][1]

    st.markdown("###  Resumen del estudiante")
    st.write(f"""
    - **Facultad:** {facultad}  
    - **Carrera:** {carrera}  
    - **Promedio:** {promedio}  
    - **Asistencia:** {asistencia}%  
    - **Nivel:** {nivel}
    """)

    if probabilidad < 0.45:
        st.success(f"🟢 Riesgo BAJO ({probabilidad*100:.1f}%)")
    elif probabilidad < 0.65:
        st.warning(f"🟡 Riesgo MEDIO ({probabilidad*100:.1f}%)")
    else:
        st.error(f"🔴 Riesgo ALTO ({probabilidad*100:.1f}%)")

st.divider()

# =========================
# IMPORTANCIA DE VARIABLES
# =========================
st.subheader("Variables más influyentes en la predicción")

coeficientes = modelo.named_steps["classifier"].coef_[0]
variables = modelo.named_steps["preprocess"].get_feature_names_out()

importancia_df = pd.DataFrame({
    "Variable": variables,
    "Importancia": coeficientes
}).sort_values(by="Importancia", ascending=False)

top_importancia = importancia_df.head(10)

st.dataframe(top_importancia)

fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
ax_imp.barh(top_importancia["Variable"], top_importancia["Importancia"])
ax_imp.set_title("Top 10 variables más influyentes")
ax_imp.invert_yaxis()
st.pyplot(fig_imp)

st.caption(
    "Valores positivos aumentan el riesgo de deserción; "
    "valores negativos lo reducen."
)
