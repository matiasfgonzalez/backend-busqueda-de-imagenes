import os
import io
import json
import logging
from typing import List, Dict, Optional
import numpy as np
import faiss
from PIL import Image
import cv2
import uuid

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
# Almacenaremos los datos de la imagen y el índice FAISS por separado
IMAGE_METADATA: List[Dict[str, str]] = []
FAISS_INDEX: Optional[faiss.IndexFlatIP] = None
EMBEDDING_DIM = 512 # Asume una dimensión para los embeddings CLIP (ajusta si tu modelo es diferente)

def remove_grid(
    image_bytes: bytes,
    output_dir: str = "./imagenes-sin-grid",
    base_filename: str = "imagen_sin_grid",
    threshold_value: int = 200,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    hough_threshold: int = 50,
    min_line_length: int = 100,
    max_line_gap: int = 10,
    kernel_size: int = 5,
    inpaint_radius: int = 3,
) -> Image.Image:
    """
    Elimina grillas de fondo usando Canny + Hough + Inpainting.
    Todos los parámetros son ajustables para pruebas/afinamiento.
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(pil_image) # Convertir PIL Image a NumPy array (RGB)
    
    # Convertir a escala de grises para procesamiento de OpenCV
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Aplicar umbralización adaptativa para aislar el dibujo y la grilla del fondo
    # Esto ayuda a que el Canny y Hough detecten mejor las líneas si el fondo no es perfectamente blanco
    # Inverse BINARY_INV para que las líneas (objetos) sean blancas y el fondo negro
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Detección de bordes (Canny) en la imagen umbralizada
    edges = cv2.Canny(thresh, canny_threshold1, canny_threshold2, apertureSize=3)

    # Detección de líneas (Transformada de Hough)
    # Ajusta los parámetros rho (resolución de distancia), theta (resolución angular),
    # threshold (mínimo de intersecciones para ser línea), minLineLength, maxLineGap.
    # Estos parámetros son CRÍTICOS y necesitarán ajuste.
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=hough_threshold, 
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap,
    )

    # Crear una máscara para las líneas detectadas
    # Inicialmente, la máscara es negra (0)
    mask = np.zeros_like(gray, dtype=np.uint8)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Dibujar las líneas en la máscara con un grosor que cubra la grilla
            # Color blanco (255) para las líneas en la máscara
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3) # Grosor de línea 3 pixels

        # Opcional: Operaciones morfológicas para asegurar que la máscara cubra bien las líneas
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # mask = cv2.erode(mask, kernel, iterations=1) # Opcional: para limpiar pequeños ruidos

    # Realizar Inpainting: Rellenar las áreas marcadas por la máscara
    # Usamos el método INPAINT_TELEA, que suele dar buenos resultados.
    # El 3 es el radio del vecindario para el inpainting.
    processed_img_np = cv2.inpaint(img_np, mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Convertir el array NumPy procesado de nuevo a PIL Image (asegúrate de que los canales RGB estén correctos)
    processed_pil_image = Image.fromarray(processed_img_np)

     # --- Guardar resultado con nombre único ---
    os.makedirs(output_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]  # 8 caracteres aleatorios
    filename = f"{base_filename}_{unique_id}.png"
    output_path = os.path.join(output_dir, filename)
    processed_pil_image.save(output_path)

    return processed_pil_image

def initialize_image_database(embedder_instance, image_folder: str = "./example_images"):
    """
    Carga imágenes de ejemplo, preprocesa para eliminar la grilla,
    genera embeddings y construye un índice FAISS.
    """
    global FAISS_INDEX, IMAGE_METADATA, EMBEDDING_DIM

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        logger.warning(f"La carpeta '{image_folder}' no existe. Por favor, agrega imágenes antes de continuar.")
        return

    embeddings_list = []
    current_metadata = []

    logger.info("Cargando, preprocesando y embebiendo imágenes...")
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(image_folder, filename)
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                # --- Aquí se llama a la función de preprocesamiento ---
                processed_pil_image = remove_grid(image_bytes)
                
                # Generar el embedding de la imagen preprocesada
                embedding = embedder_instance.get_image_embedding(processed_pil_image)
                embeddings_list.append(embedding)
                current_metadata.append({"id": filename, "path": image_path})

                logger.info(f"Procesada: {filename}")
            except Exception as e:
                logger.error(f"Error procesando imagen {filename}: {e}")

    if not embeddings_list:
        logger.error(f"No se encontraron imágenes en '{image_folder}'.")
        return

    embeddings_array = np.array(embeddings_list).astype('float32')

    if embeddings_array.shape[1] != EMBEDDING_DIM:
        print(f"Advertencia: La dimensión del embedding ({embeddings_array.shape[1]}) no coincide con la dimensión inicial EMBEDDING_DIM ({EMBEDDING_DIM}). Ajustando EMBEDDING_DIM.")
        EMBEDDING_DIM = embeddings_array.shape[1]

    FAISS_INDEX = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss.normalize_L2(embeddings_array)
    FAISS_INDEX.add(embeddings_array)

    IMAGE_METADATA = current_metadata
    print(f"Índice FAISS construido con {len(IMAGE_METADATA)} imágenes.")


def find_similar_images(query_embedding: np.ndarray, top_n: Optional[int] = 5) -> List[Dict[str, any]]:
    """
    Encuentra las top N imágenes más similares usando el índice FAISS.
    """
    global FAISS_INDEX, IMAGE_METADATA, EMBEDDING_DIM

    if FAISS_INDEX is None or not IMAGE_METADATA:
        print("Error: El índice FAISS no está inicializado o no hay metadatos.")
        return []

    query_embedding_normalized = query_embedding.astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding_normalized)

    k = top_n if top_n is not None else FAISS_INDEX.ntotal
    distances, indices = FAISS_INDEX.search(query_embedding_normalized, k)

    results = []
    for i, sim in zip(indices[0], distances[0]):
        if i == -1:
            continue
        if i < len(IMAGE_METADATA):
            metadata = IMAGE_METADATA[i]
            results.append({
                "id": metadata["id"],
                "path": metadata["path"],
                "similarity": sim
            })

    return results

def save_index(index_path: str = "./faiss_index.index", metadata_path: str = "./metadata.json"):
    """
    Guarda el índice FAISS y los metadatos en disco.
    """
    global FAISS_INDEX, IMAGE_METADATA

    if FAISS_INDEX is None or not IMAGE_METADATA:
        logger.error("No hay índice o metadata para guardar.")
        return

    try:
        faiss.write_index(FAISS_INDEX, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(IMAGE_METADATA, f, indent=2, ensure_ascii=False)
        logger.info(f"Índice guardado en {index_path}, metadata en {metadata_path}")
    except Exception as e:
        logger.error(f"Error guardando índice: {e}")


def load_index(index_path: str = "./faiss_index.index", metadata_path: str = "./metadata.json"):
    """
    Carga el índice FAISS y metadata desde disco.
    """
    global FAISS_INDEX, IMAGE_METADATA

    try:
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            FAISS_INDEX = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                IMAGE_METADATA = json.load(f)
            logger.info(f"Índice cargado desde {index_path}, metadata desde {metadata_path}")
        else:
            logger.warning("No se encontró índice o metadata para cargar.")
    except Exception as e:
        logger.error(f"Error cargando índice: {e}")