from typing import List, Dict, Optional, Tuple
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity # Ya no la usaremos para la búsqueda principal
from PIL import Image
import os
import io
import faiss # Importa FAISS

# Almacenaremos los datos de la imagen y el índice FAISS por separado
# IMAGE_METADATA contendrá el ID y la ruta (y cualquier otro metadato)
IMAGE_METADATA: List[Dict[str, str]] = []
FAISS_INDEX: Optional[faiss.IndexFlatIP] = None # Usaremos IndexFlatIP para similitud de coseno
EMBEDDING_DIM = 512 # Asume una dimensión para los embeddings CLIP (ajusta si tu modelo es diferente)

def initialize_image_database(embedder_instance):
    """
    Carga imágenes de ejemplo, genera embeddings y construye un índice FAISS.
    """
    global FAISS_INDEX, IMAGE_METADATA, EMBEDDING_DIM # Para modificar las variables globales

    image_folder = "./example_images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"La carpeta '{image_folder}' no existe. Por favor, agrega algunas imágenes aquí.")
        return

    embeddings_list = []
    current_metadata = []

    print("Cargando y embebiendo imágenes para el índice FAISS...")
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(image_folder, filename)
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                embedding = embedder_instance.get_image_embedding(img)
                embeddings_list.append(embedding)
                current_metadata.append({"id": filename, "path": image_path}) # Solo metadatos ligeros

                print(f"Procesada: {filename}")
            except Exception as e:
                print(f"Error procesando imagen {filename}: {e}")

    if not embeddings_list:
        print(f"No se encontraron imágenes en '{image_folder}'. No se pudo construir el índice FAISS.")
        return

    # Convertir la lista de embeddings a un array NumPy
    # Asegúrate de que los embeddings son de tipo float32, que es lo que FAISS espera
    embeddings_array = np.array(embeddings_list).astype('float32')

    # Verificar la dimensión de los embeddings y ajustarla si es necesario
    if embeddings_array.shape[1] != EMBEDDING_DIM:
        print(f"Advertencia: La dimensión del embedding ({embeddings_array.shape[1]}) no coincide con la dimensión inicial EMBEDDING_DIM ({EMBEDDING_DIM}). Ajustando EMBEDDING_DIM.")
        EMBEDDING_DIM = embeddings_array.shape[1] # Esto ya está dentro del ámbito global gracias a la declaración


    # Construir el índice FAISS
    # IndexFlatIP: Para búsqueda de productos internos (equivalente a similitud de coseno si los vectores están normalizados)
    FAISS_INDEX = faiss.IndexFlatIP(EMBEDDING_DIM)
    # Normalizar los embeddings para que el producto interno sea igual a la similitud de coseno
    faiss.normalize_L2(embeddings_array)
    FAISS_INDEX.add(embeddings_array) # Añadir los embeddings al índice

    IMAGE_METADATA = current_metadata # Guardar solo los metadatos
    print(f"Índice FAISS construido con {len(IMAGE_METADATA)} imágenes.")


def find_similar_images(query_embedding: np.ndarray, top_n: Optional[int] = 5) -> List[Dict[str, any]]:
    """
    Encuentra las top N imágenes más similares usando el índice FAISS.
    """
    if FAISS_INDEX is None or not IMAGE_METADATA:
        print("Error: El índice FAISS no está inicializado o no hay metadatos.")
        return []

    # Normalizar el embedding de la consulta antes de buscar en FAISS
    query_embedding_normalized = query_embedding.astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding_normalized)

    # Realizar la búsqueda en FAISS
    # D: Distancias (similitudes), I: Índices de los vecinos más cercanos
    k = top_n if top_n is not None else FAISS_INDEX.ntotal # Si top_n es None, buscar todos
    distances, indices = FAISS_INDEX.search(query_embedding_normalized, k)

    results = []
    for i, sim in zip(indices[0], distances[0]):
        if i == -1: # FAISS devuelve -1 si no encuentra suficientes resultados
            continue
        # Asegurarse de que el índice esté dentro del rango de los metadatos
        if i < len(IMAGE_METADATA):
            metadata = IMAGE_METADATA[i]
            results.append({
                "id": metadata["id"],
                "path": metadata["path"],
                "similarity": sim # FAISS devuelve la similitud directamente
            })

    # Asegúrate de que los resultados estén ordenados de mayor a menor similitud (FAISS ya lo hace por defecto)
    # y aplicar el umbral aquí si lo deseas, o déjalo en el main.py
    # En este caso, el filtrado por umbral se seguirá haciendo en main.py

    return results