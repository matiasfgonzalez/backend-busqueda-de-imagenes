from typing import List, Dict
import numpy as np
from PIL import Image
import os
import logging

from .database import SessionLocal, ImageEmbedding
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Nota: ya no usamos IMAGE_DATABASE en memoria
# sino que trabajamos contra la tabla image_embeddings en Postgres + pgvector.

def initialize_image_database(embedder_instance, image_folder: str = "./example_images"):
    """
    Recorre la carpeta de ejemplo, genera embeddings y los guarda en la base de datos
    si no existen todavía en la tabla image_embeddings.
    """
    # Aseguramos la carpeta
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"La carpeta '{image_folder}' no existía. Créala y agregá imágenes para poblar la DB.")
        return

    with SessionLocal() as session:
        # Contar cuántas filas hay en la tabla
        count = session.query(ImageEmbedding).count()
        if count > 0:
            print(f"Ya existen {count} embeddings en la base de datos. Se omite la carga inicial.")
            return

        # Si no hay embeddings, los generamos desde archivos en la carpeta
        images_to_add = []
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')):
                image_path = os.path.join(image_folder, filename)
                try:
                    img = Image.open(image_path).convert("RGB")
                    embedding = embedder_instance.get_image_embedding(img)
                    # Convertir a lista de floats para que psycopg2/pgvector lo acepte
                    emb_list = [float(x) for x in embedding.tolist()]
                    
                    # Validar dimensión del vector
                    if len(emb_list) != 512:
                        print(f"Error: embedding de {filename} tiene {len(emb_list)} dimensiones, se esperaban 512")
                        continue

                    new_row = ImageEmbedding(
                        image_path=f"/static/{filename}",  # Path relativo para el frontend
                        embedding=emb_list
                    )
                    images_to_add.append(new_row)
                    print(f"Preparada imagen: {filename}")
                except Exception as e:
                    print(f"Error procesando imagen {filename}: {e}")
        
        # Agregar todas las imágenes en un solo commit
        if images_to_add:
            try:
                session.add_all(images_to_add)
                session.commit()
                print(f"Inicialización completa: {len(images_to_add)} embeddings almacenados.")
            except Exception as e:
                session.rollback()
                print(f"Error guardando embeddings: {e}")
        else:
            print(f"No se encontraron imágenes en '{image_folder}' o todas fallaron al procesar.")


def find_similar_images(query_embedding: np.ndarray, top_n: int = 5) -> List[Dict[str, any]]:
    """
    Busca en la base de datos las top N imágenes más similares usando pgvector.
    Devuelve lista de dicts con id, path y similarity (0..1).
    """
    if query_embedding is None:
        return []

    # Convertir a lista de floats y luego a string en formato pgvector
    vector_param = [float(x) for x in query_embedding.tolist()]
    vector_str = '[' + ','.join(map(str, vector_param)) + ']'

    with SessionLocal() as session:
        # Usamos SQL con parámetros bind para evitar SQL injection
        sql = text("""
            SELECT id, image_path, embedding <-> :embedding::vector AS distance
            FROM image_embeddings
            ORDER BY distance
            LIMIT :limit
        """)
        result = session.execute(sql, {"embedding": vector_str, "limit": top_n}).fetchall()

        logger.info(f"Query ejecutada. Resultados encontrados: {len(result)}")

        # Convertimos distancia a similaridad
        # Para vectores normalizados, la distancia L2 está en el rango [0, 2]
        # donde 0 = idénticos y 2 = completamente opuestos
        items = []
        for row in result:
            row_id = row[0]
            path = row[1]
            distance = float(row[2])
            
            # Convertir distancia L2 a similitud (0..1)
            # Para vectores normalizados: similarity = 1 - (distance / 2)
            # O alternativamente: similarity = 1 - (distance^2 / 4)
            # Usamos la fórmula más simple
            similarity = max(0.0, 1.0 - (distance / 2.0))
            
            logger.debug(f"ID: {row_id}, Path: {path}, Distance: {distance:.4f}, Similarity: {similarity:.4f}")
            items.append({"id": row_id, "path": path, "similarity": similarity, "distance": distance})
        
        return items