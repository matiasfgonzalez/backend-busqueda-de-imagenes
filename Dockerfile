# Usa una imagen base de Python
FROM python:3.10-slim-buster

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requisitos e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY app/ ./app/
COPY example_images/ ./example_images/

# Expone el puerto que usará FastAPI
EXPOSE 8000

# Comando para correr la aplicación con Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]