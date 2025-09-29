# Usa una imagen base de Python
# FROM python:3.10-slim-buster
FROM python:3.10-slim-bullseye

# Establece el directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema necesarias para OpenCV
# Estas son bibliotecas que OpenCV podría necesitar incluso si no usas una GUI
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos de requisitos e instala las dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY app/ ./app/
COPY example_images/ ./example_images/
COPY imagenes-sin-grid/ ./imagenes-sin-grid/

# Expone el puerto que usará FastAPI
EXPOSE 8000

# Comando para correr la aplicación con Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]