from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

# Simulamos una base de datos de imágenes.
# En un proyecto real, esto se cargaría de un almacenamiento persistente
# y los embeddings se generarían una sola vez o se cargarían desde un índice.
IMAGE_DATABASE: List[Dict[str, any]] = []

def initialize_image_database(embedder_instance):
    """
    Carga imágenes de ejemplo desde una carpeta y genera sus embeddings.
    Este es un ejemplo simple. En producción, cargarías de un almacenamiento
    y los embeddings ya deberían estar precalculados.
    """
    image_folder = "./example_images" # Asegúrate de que esta carpeta exista y contenga imágenes
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"La carpeta '{image_folder}' no existe. Por favor, agrega algunas imágenes aquí.")
        return

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(image_folder, filename)
            try:
                img = Image.open(image_path).convert("RGB")
                embedding = embedder_instance.get_image_embedding(img)
                IMAGE_DATABASE.append({
                    "id": filename, # Usamos el nombre de archivo como ID
                    "path": image_path,
                    "embedding": embedding
                })
                print(f"Cargada y embebida imagen: {filename}")
            except Exception as e:
                print(f"Error procesando imagen {filename}: {e}")
    if not IMAGE_DATABASE:
        print(f"No se encontraron imágenes en '{image_folder}'. Por favor, agrega algunas imágenes de ejemplo.")


def find_similar_images(query_embedding: np.ndarray, top_n: int = 5) -> List[Dict[str, any]]:
    """
    Encuentra las top N imágenes más similares a partir de la base de datos.
    """
    if not IMAGE_DATABASE:
        return []

    similarities = []
    for item in IMAGE_DATABASE:
        db_embedding = item["embedding"]
        # Reshape para cosine_similarity: (n_samples, n_features)
        similarity = cosine_similarity(query_embedding.reshape(1, -1), db_embedding.reshape(1, -1))[0][0]
        similarities.append({"id": item["id"], "path": item["path"], "similarity": similarity})

    # Ordenar por similitud de forma descendente
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    # Devolver las top N, excluyendo la propia imagen si está en la base de datos (si aplica)
    # Por simplicidad, no excluimos explícitamente la imagen de consulta si ya está en la DB.
    # En un caso real, si la imagen subida es de la DB, la omitiríamos.
    return similarities[:top_n]