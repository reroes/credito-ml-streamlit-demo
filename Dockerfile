# Imagen base: Python
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivo requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación y los modelos
COPY app.py .
COPY feature_names.pkl .
COPY modelo_credito_svm.pkl .

# Exponer puerto de Streamlit
EXPOSE 8501

# Ejecutar Streamlit al iniciar el contenedor
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
