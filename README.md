# 🔎 Backend de Búsqueda de Imágenes con CLIP + FAISS

Este proyecto es una **API backend construida con FastAPI** que permite buscar imágenes similares a partir de una imagen de referencia.  
Utiliza **CLIP (OpenAI)** para generar embeddings de imágenes y **FAISS (Facebook AI Similarity Search)** para realizar búsquedas eficientes de similitud.  
Además, incluye un preprocesamiento con **OpenCV** para eliminar grillas de fondo en las imágenes (útil para trabajar con dibujos o imágenes escaneadas).

---

## ✨ Características

- API REST construida con **FastAPI**.
- Extracción de embeddings con **CLIP (transformers de Hugging Face)**.
- Búsqueda de similitud usando **FAISS**.
- Preprocesamiento de imágenes con **OpenCV** para remover grillas de fondo.
- Soporte para **CORS** (para integrarlo con un frontend en React/Next.js).
- Servidor estático de imágenes de ejemplo (`/static`).

---

## 🛠️ Tecnologías utilizadas

- [FastAPI](https://fastapi.tiangolo.com/) → framework web en Python.
- [Transformers (HuggingFace)](https://huggingface.co/) → modelo CLIP para embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) → motor de búsqueda de similitud.
- [OpenCV](https://opencv.org/) → preprocesamiento y limpieza de imágenes.
- [PIL (Pillow)](https://python-pillow.org/) → manejo de imágenes.
- [Docker](https://www.docker.com/) → ejecución en contenedores.

---

## 📂 Estructura del proyecto

```bash
    ├── app/
    │ ├── main.py # Definición de la API con FastAPI
    │ ├── model.py # Clase ImageEmbedder con CLIP
    │ ├── utils.py # Funciones de FAISS y preprocesamiento de imágenes
    ├── example_images/ # Imágenes de ejemplo para inicializar la BD
    ├── test_images/ # Imágenes para testing manual
    ├── requirements.txt # Dependencias de Python
    ├── Dockerfile # Soporte para ejecución en Docker
    ├── .env.template # Variables de entorno de ejemplo
    └── README.md # Documentación del proyecto
```

## ⚙️ Instalación

### 🔹 Opción 1: Local (con Python)

1. Clonar el repositorio:

```bash
    git clone https://github.com/matiasfgonzalez/backend-busqueda-de-imagenes.git
    cd backend-busqueda-de-imagenes
```

2. Crear un entorno virtual e instalar dependencias:

```bash
    python -m venv venv
    source venv/bin/activate    # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
```

3. Configurar variables de entorno:

```bash
    cp .env.template .env
    # Editar .env según corresponda
```

4. Ejecutar el servidor:

```bash
    uvicorn app.main:app --reload
```

La API estará disponible en: http://localhost:8000

### 🔹 Opción 2: Con Docker

```bash
    docker build -t image-search-backend .
    docker run -p 8000:8000 --env-file .env image-search-backend
```

### 🌍 Endpoints principales

POST /search-similar-images/

Busca imágenes similares a una imagen de referencia.

Ejemplo con curl:

```bash
    curl -X POST "http://localhost:8000/search-similar-images/" \
    -F "file=@test_images/ejemplo.jpg"
```

Respuesta:

```bash
    {
    "results": [
        {
        "id": "imagen1.jpg",
        "similarity": 0.92,
        "path": "./example_images/imagen1.jpg"
        },
        {
        "id": "imagen2.jpg",
        "similarity": 0.89,
        "path": "./example_images/imagen2.jpg"
        }
    ]
    }
```

### 📌 Detalles técnicos

- Embeddings: generados con openai/clip-vit-base-patch32.

- Dimensión del embedding: 512 (ajustable según el modelo).

- Similitud: se usa búsqueda basada en cosine similarity (normalización L2 en FAISS).

- Preprocesamiento: las imágenes pasan por remove_grid para detectar y eliminar grillas con OpenCV.

### 🧪 Tests

Podés probar subiendo imágenes a example_images/ y consultando con test_images/.

El índice FAISS se construye automáticamente en el arranque con las imágenes de example_images.

### 🤝 Contribuciones

1. Las contribuciones son bienvenidas.

2. Haz un fork del proyecto.

3. Crea una rama (git checkout -b feature-nueva).

4. Haz commit de tus cambios (git commit -m 'Agrego nueva feature').

5. Haz push a la rama (git push origin feature-nueva).

6. Abre un Pull Request.

### 📄 Licencia

MIT License. Libre para uso y modificación.
