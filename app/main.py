from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
import cv2 # Importa OpenCV
import numpy as np # Necesario para cv2

from .model import image_embedder
from .utils import find_similar_images, initialize_image_database, remove_grid # ¡Importa remove_grid!

app = FastAPI(
    title="Generative AI Image Search Backend",
    description="API para buscar imágenes similares usando embeddings de CLIP.",
    version="0.1.0"
)

# Configuración de CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta la carpeta 'example_images' para servir archivos estáticos
app.mount("/static", StaticFiles(directory="./example_images"), name="static")

@app.on_event("startup")
async def startup_event():
    print("Inicializando base de datos de imágenes...")
    os.makedirs("./example_images", exist_ok=True)
    initialize_image_database(image_embedder)
    # La advertencia de "No se cargaron imágenes" ahora está en initialize_image_database

@app.post("/search-similar-images/")
async def search_similar_images(file: UploadFile = File(...)):
    """
    Endpoint para buscar imágenes similares a partir de una imagen de referencia.
    La imagen de referencia será preprocesada para eliminar la grilla.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen.")

    try:
        # Leer la imagen subida en bytes
        contents = await file.read()
        
        # --- Preprocesar la imagen de consulta para eliminar la grilla ---
        processed_query_image = remove_grid(contents)

        # Generar el embedding de la imagen de consulta preprocesada
        query_embedding = image_embedder.get_image_embedding(processed_query_image)

        # Buscar imágenes similares
        similar_images = find_similar_images(query_embedding, top_n=15)

        # Filtra los resultados para incluir solo aquellos con similitud >= 0.8
        results_for_frontend = [
            {"id": img["id"], "similarity": float(img["similarity"]), "path": img["path"]}
            for img in similar_images
            if img["similarity"] >= 0.9
        ]

        return JSONResponse(content={"results": results_for_frontend})

    except Exception as e:
        print(f"Error procesando la solicitud: {e}")
        # En un entorno de producción, es mejor no exponer directamente el detalle del error al cliente.
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la imagen.")