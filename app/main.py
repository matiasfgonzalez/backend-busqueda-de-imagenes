from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os

from .model import image_embedder # Importa la instancia global
from .utils import find_similar_images, initialize_image_database, IMAGE_DATABASE

app = FastAPI(
    title="Generative AI Image Search Backend",
    description="API para buscar imágenes similares usando embeddings de CLIP.",
    version="0.1.0"
)

# Configuración de CORS para permitir solicitudes desde el frontend de Next.js
origins = [
    "http://localhost:3000",  # Frontend Next.js local
    # Puedes añadir otros orígenes aquí si tu frontend se despliega en otro lugar
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ¡Añade esta línea para servir archivos estáticos! ---
# 'directory="./example_images"' es la carpeta donde FastAPI buscará los archivos.
# 'name="static"' es un nombre interno para la ruta, no la URL.
app.mount("/static", StaticFiles(directory="./example_images"), name="static")

@app.on_event("startup")
async def startup_event():
    print("Inicializando base de datos de imágenes...")
    # Asegúrate de crear una carpeta 'example_images' en el directorio 'backend'
    # y colocar algunas imágenes allí para probar.
    os.makedirs("./example_images", exist_ok=True)
    initialize_image_database(image_embedder)
    if not IMAGE_DATABASE:
        print("Advertencia: No se cargaron imágenes en la base de datos. Asegúrate de tener imágenes en 'backend/example_images'.")


@app.post("/search-similar-images/")
async def search_similar_images(file: UploadFile = File(...)):
    """
    Endpoint para buscar imágenes similares a partir de una imagen de referencia.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen.")

    try:
        # Leer la imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Generar el embedding de la imagen de consulta
        query_embedding = image_embedder.get_image_embedding(image)

        # Buscar imágenes similares
        similar_images = find_similar_images(query_embedding, top_n=5) # Ajusta top_n según necesidad

        # Retornar los resultados (ID, similitud, y quizás la URL de la imagen si existiera en un CDN)
        # Aquí, para fines de demostración, solo retornamos el ID y la similitud.
        # En un escenario real, el frontend necesitaría una URL para mostrar la imagen.
        # Por ahora, asumimos que el frontend puede construir la URL si tiene el ID/nombre.
        results_for_frontend = [
            {"id": img["id"], "similarity": float(img["similarity"]), "path": img["path"]}
            for img in similar_images
            if img["similarity"] >= 0.8
        ]

        return JSONResponse(content={"results": results_for_frontend})

    except Exception as e:
        print(f"Error procesando la solicitud: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")