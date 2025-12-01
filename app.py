import pandas as pd
import joblib
import streamlit as st

# =========================
# 1. Cargar modelo + features
# =========================
modelo = joblib.load("modelo_credito_svm.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# 2. UI Streamlit
# =========================
st.title("Demo ML: Aprobación de Crédito (SVM + StandardScaler) + GitHub")
st.write("Modelo entrenado en notebook y cargado desde archivo .pkl.")

st.sidebar.header("Datos del cliente")

ingreso_input = st.sidebar.number_input(
    "Ingreso mensual (USD)", min_value=500, max_value=20000, value=4000, step=100)

antiguedad_input = st.sidebar.number_input(
    "Antigüedad laboral (años)", min_value=0, max_value=40, value=5, step=1)

edad_input = st.sidebar.number_input(
    "Edad", min_value=18, max_value=80, value=30, step=1)

if st.sidebar.button("Evaluar solicitud"):
    nuevo = pd.DataFrame([{
        feature_names[0]: ingreso_input,
        feature_names[1]: antiguedad_input,
        feature_names[2]: edad_input
    }])[feature_names]  # respeta el orden exacto

    pred = modelo.predict(nuevo)[0]
    proba = modelo.predict_proba(nuevo)[0][1]

    st.subheader("Resultado del modelo")
    if pred == 1:
        st.success(f"✅ Crédito APROBADO (prob. de aprobación: {proba:.1%})")
    else:
        st.error(f"❌ Crédito NO aprobado (prob. de aprobación: {proba:.1%})")
