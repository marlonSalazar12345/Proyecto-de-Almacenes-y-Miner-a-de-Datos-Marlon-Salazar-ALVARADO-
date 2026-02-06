# Proyecto-de-Almacenes-y-Miner-a-de-Datos-Marlon-Salazar-ALVARADO-
# PROYECTO DE MINERÍA DE DATOS

## Predicción de Deserción Estudiantil

**Autor:** Allison Castro
**Asignatura:** Minería de Datos
**Herramientas:** Python, Pandas, Scikit-learn, Streamlit

---

## Descripción del Proyecto

La deserción estudiantil representa uno de los principales desafíos que enfrentan las instituciones de educación superior, ya que impacta negativamente en la planificación académica, el uso de recursos y los indicadores de calidad educativa. Identificar de manera temprana a los estudiantes con riesgo de abandono permite implementar estrategias de intervención oportunas.

En este proyecto se aplican técnicas de minería de datos para desarrollar un modelo predictivo que identifique estudiantes con riesgo de deserción a partir de información académica histórica. Los resultados del análisis y del modelo predictivo se presentan mediante una aplicación interactiva desarrollada con Streamlit, facilitando la visualización y la toma de decisiones.

---

## Objetivo General

Desarrollar un sistema de predicción de deserción estudiantil aplicando técnicas de minería de datos, que permita identificar estudiantes en riesgo y visualizar los resultados mediante una interfaz gráfica interactiva.

---

## Objetivos Específicos

* Realizar un análisis exploratorio del conjunto de datos académico.
* Identificar las variables más relevantes para la predicción de la deserción estudiantil.
* Definir la variable objetivo a partir de los registros históricos.
* Aplicar técnicas de limpieza y preprocesamiento de datos.
* Construir y evaluar un modelo de clasificación.
* Desarrollar una aplicación interactiva utilizando Streamlit.
* Documentar todo el proceso siguiendo la metodología CRISP-DM.

---

## 🗂️ Dataset

El proyecto utiliza un archivo Excel anonimizado proporcionado por la institución:

* **REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx**

Principales variables utilizadas:

* **PROMEDIO:** promedio académico del estudiante.
* **ASISTENCIA:** porcentaje de asistencia a clases.
* **NIVEL:** nivel académico cursado.
* **FACULTAD:** facultad a la que pertenece el estudiante.
* **CARRERA:** carrera del estudiante.
* **ESTADO:** estado del estudiante (variable objetivo: deserta / no deserta).

---

## Metodología CRISP-DM

El desarrollo del proyecto sigue la metodología **CRISP-DM**, abordando las siguientes fases:

### 1. Comprensión del negocio

El objetivo institucional es reducir la deserción estudiantil mediante la identificación temprana de estudiantes en riesgo. La información generada por el modelo permitirá implementar acciones preventivas como tutorías académicas, seguimiento personalizado y apoyo institucional.

### 2. Comprensión de los datos

Se realizó un análisis exploratorio de los datos (EDA) para comprender su estructura, identificar valores nulos, analizar distribuciones y observar la proporción de estudiantes que desertan y no desertan.

### 3. Preparación de los datos

En esta fase se llevaron a cabo las siguientes tareas:

* Limpieza de valores nulos.
* Normalización de los nombres de las columnas.
* Selección de variables relevantes.
* Codificación de variables categóricas (facultad y carrera).
* Separación de variables predictoras y variable objetivo.

### 4. Modelado

Se implementó un modelo de **Regresión Logística**, utilizando un pipeline que integra el preprocesamiento de los datos y el algoritmo de clasificación. Este modelo fue seleccionado por su simplicidad, interpretabilidad y buen desempeño en problemas de clasificación binaria.

### 5. Evaluación

El modelo fue evaluado utilizando las siguientes métricas:

* Accuracy (Exactitud)
* Precision (Precisión)
* Recall (Sensibilidad)
* F1-score
* Matriz de confusión

Los resultados muestran un desempeño aceptable del modelo, destacando un valor alto de *recall*, lo cual es especialmente importante para identificar a la mayoría de estudiantes en riesgo de deserción.

### 6. Despliegue

El modelo fue desplegado mediante una aplicación interactiva desarrollada con **Streamlit**, la cual permite:

* Visualizar el análisis exploratorio del dataset.
* Mostrar estadísticas descriptivas.
* Evaluar el desempeño del modelo.
* Ingresar los datos académicos de un estudiante.
* Obtener la predicción del riesgo de deserción en tiempo real.
* Visualizar la importancia de las variables utilizadas por el modelo.

---

## Modelo de Machine Learning

Se implementó un modelo de **Regresión Logística** para la predicción de la deserción estudiantil, entrenado a partir de variables académicas y administrativas.

### Criterio de Riesgo

El riesgo de deserción se determina a partir de la probabilidad estimada por el modelo:

* **Riesgo bajo:** probabilidad menor al 45%
* **Riesgo medio:** probabilidad entre 45% y 65%
* **Riesgo alto:** probabilidad mayor al 65%

Este enfoque permite una clasificación más flexible y realista que el uso de reglas fijas.

---

## 📈 Evaluación del Modelo

El desempeño del modelo se presenta en la aplicación mediante:

* Métricas de clasificación (accuracy, precision, recall y F1-score).
* Visualización de la matriz de confusión.

Estas herramientas permiten analizar el comportamiento del modelo y su capacidad para identificar correctamente a los estudiantes en riesgo.

---

## Aplicación Streamlit

La aplicación desarrollada con Streamlit permite:

* Explorar visualmente los datos académicos.
* Analizar la distribución del estado de los estudiantes.
* Ingresar datos académicos de un estudiante.
* Obtener la predicción del riesgo de deserción en tiempo real.

Para ejecutar la aplicación:

```bash
streamlit run app.py
```

---

## Conclusiones

El proyecto demuestra que el uso de técnicas de minería de datos y la metodología CRISP-DM permiten abordar de manera efectiva el problema de la deserción estudiantil. El modelo desarrollado constituye una herramienta de apoyo para la toma de decisiones académicas y la implementación de estrategias preventivas.

Como trabajo futuro, se recomienda incorporar nuevas variables, probar otros algoritmos de clasificación y actualizar periódicamente el modelo con nuevos datos.
