from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from PIL import Image
import io
import os
import logging

from .model import image_embedder  # instancia global
from .utils import find_similar_images, initialize_image_database
from .database import create_tables, SessionLocal
from sqlalchemy import text

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager para manejar el ciclo de vida de la aplicación.
    """
    # Startup
    logger.info("Inicializando base de datos y creando tablas si es necesario...")
    try:
        create_tables()
        os.makedirs("./example_images", exist_ok=True)
        initialize_image_database(image_embedder, image_folder="./example_images")
        logger.info("Inicialización completada exitosamente")
    except Exception as e:
        logger.error(f"Error durante la inicialización: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación...")


app = FastAPI(
    title="Generative AI Image Search Backend",
    description="API para buscar imágenes similares usando embeddings de CLIP.",
    version="0.1.0",
    lifespan=lifespan
)

# Configuración de CORS para permitir solicitudes desde el frontend de Next.js
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (imágenes de ejemplo)
app.mount("/static", StaticFiles(directory="./example_images"), name="static")


@app.get("/health")
async def health_check():
    """
    Endpoint de health check para verificar el estado del servicio y la base de datos.
    """
    try:
        # Verificar conexión a base de datos
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "service": "image-search-backend",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check falló: {e}")
        raise HTTPException(status_code=503, detail=f"Servicio no disponible: {str(e)}")


@app.post("/search-similar-images/")
async def search_similar_images(file: UploadFile = File(...)):
    """
    Endpoint para buscar imágenes similares a partir de una imagen de referencia.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen.")

    try:
        # Leer la imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Generar el embedding de la imagen de consulta
        logger.info(f"Generando embedding para imagen: {file.filename}")
        query_embedding = image_embedder.get_image_embedding(image)
        
        # Validar dimensión del embedding
        if len(query_embedding) != 512:
            raise ValueError(f"Embedding generado tiene {len(query_embedding)} dimensiones, se esperaban 512")

        # Buscar imágenes similares
        similar_images = find_similar_images(query_embedding, top_n=10)
        
        logger.info(f"Imágenes encontradas antes de filtrar: {len(similar_images)}")
        if similar_images:
            logger.info(f"Mejor coincidencia - ID: {similar_images[0]['id']}, "
                       f"Similarity: {similar_images[0]['similarity']:.4f}, "
                       f"Distance: {similar_images[0]['distance']:.4f}")

        # Filtrar por un umbral de similitud (0.5 = 50% de similitud)
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
        
        # Construir URL base para las imágenes
        # En producción, esto debería venir de una variable de entorno
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        
        results_for_frontend = [
            {
                "id": img["id"], 
                "similarity": float(img["similarity"]), 
                "path": f"{base_url}{img['path']}",  # URL completa
                "distance": float(img["distance"])
            }
            for img in similar_images
            if img["similarity"] >= threshold
        ]
        
        logger.info(f"Encontradas {len(results_for_frontend)} imágenes con similitud >= {threshold}")
        return JSONResponse(content={"results": results_for_frontend, "threshold": threshold})

    except Exception as e:
        logger.error(f"Error procesando la solicitud: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
