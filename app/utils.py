from typing import List, Dict
import numpy as np
from PIL import Image
import os
import hashlib
import logging
import threading
import uuid

from .database import SessionLocal, ImageEmbedding
from .model import TOTAL_DIM
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Lock para garantizar concurrencia segura al insertar nuevas imágenes
_index_lock = threading.Lock()

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
                    embedding = embedder_instance.get_combined_embedding(img)
                    # Convertir a lista de floats para que psycopg2/pgvector lo acepte
                    emb_list = [float(x) for x in embedding.tolist()]
                    
                    # Validar dimensión del vector
                    if len(emb_list) != TOTAL_DIM:
                        print(f"Error: embedding de {filename} tiene {len(emb_list)} dimensiones, se esperaban {TOTAL_DIM}")
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
            SELECT id, image_path, original_filename, embedding <-> CAST(:embedding AS vector) AS distance
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
            original_filename = row[2]
            distance = float(row[3])
            
            # Convertir distancia L2 a similitud (0..1)
            # Para vectores normalizados: similarity = 1 - (distance / 2)
            # O alternativamente: similarity = 1 - (distance^2 / 4)
            # Usamos la fórmula más simple
            similarity = max(0.0, 1.0 - (distance / 2.0))
            
            logger.debug(f"ID: {row_id}, Path: {path}, Distance: {distance:.4f}, Similarity: {similarity:.4f}")
            items.append({"id": row_id, "path": path, "original_filename": original_filename, "similarity": similarity, "distance": distance})
        
        return items


# ---------------------------------------------------------------------------
# Constantes para validación de imágenes subidas
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
UPLOAD_FOLDER = "./uploaded_images"


def compute_sha256(data: bytes) -> str:
    """Calcula el hash SHA-256 de los bytes de un archivo."""
    return hashlib.sha256(data).hexdigest()


def validate_image_file(filename: str, file_size: int) -> None:
    """
    Valida nombre de archivo y tamaño.
    Lanza ValueError si la validación falla.
    """
    if not filename:
        raise ValueError("El archivo no tiene nombre.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Extensión '{ext}' no permitida. Formatos aceptados: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ValueError(f"El archivo excede el tamaño máximo permitido de {max_mb:.0f} MB.")


def add_image_to_database(
    embedder_instance,
    image_bytes: bytes,
    original_filename: str,
) -> Dict:
    """
    Persiste una imagen en el filesystem, genera su embedding e inserta
    el registro en la base de datos de forma atómica.

    Args:
        embedder_instance: Instancia de ImageEmbedder.
        image_bytes: Contenido raw del archivo de imagen.
        original_filename: Nombre original del archivo subido.

    Returns:
        Dict con id, image_path y sha256_hash de la imagen creada.

    Raises:
        ValueError: Validación de archivo.
        RuntimeError: Error de persistencia / indexación.
    """
    # --- 1. Validar archivo ---
    validate_image_file(original_filename, len(image_bytes))

    # --- 2. Calcular hash para deduplicación ---
    file_hash = compute_sha256(image_bytes)

    with _index_lock:
        # Verificar duplicado dentro del lock para evitar race condition
        with SessionLocal() as session:
            existing = session.query(ImageEmbedding).filter_by(sha256_hash=file_hash).first()
            if existing:
                raise ValueError(
                    f"La imagen ya existe en la base de datos (ID: {existing.id}, "
                    f"path: {existing.image_path})."
                )

        # --- 3. Abrir y validar la imagen con Pillow ---
        try:
            image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"No se pudo abrir la imagen: {e}")

        # --- 4. Generar embedding ---
        try:
            embedding = embedder_instance.get_combined_embedding(image)
            emb_list = [float(x) for x in embedding.tolist()]
            if len(emb_list) != TOTAL_DIM:
                raise RuntimeError(
                    f"Embedding generado tiene {len(emb_list)} dimensiones, se esperaban {TOTAL_DIM}"
                )
        except Exception as e:
            raise RuntimeError(f"Error generando embedding: {e}")

        # --- 5. Guardar archivo en filesystem ---
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        ext = os.path.splitext(original_filename)[1].lower()
        safe_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        try:
            with open(file_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            raise RuntimeError(f"Error guardando archivo en disco: {e}")

        # --- 6. Insertar en base de datos (atómico con rollback) ---
        relative_path = f"/uploads/{safe_filename}"
        try:
            with SessionLocal() as session:
                new_row = ImageEmbedding(
                    image_path=relative_path,
                    embedding=emb_list,
                    sha256_hash=file_hash,
                    original_filename=original_filename,
                )
                session.add(new_row)
                session.commit()
                session.refresh(new_row)

                created_id = new_row.id
                logger.info(
                    f"Imagen indexada exitosamente – ID: {created_id}, "
                    f"path: {relative_path}, hash: {file_hash[:16]}…"
                )
                return {
                    "id": created_id,
                    "image_path": relative_path,
                    "sha256_hash": file_hash,
                    "original_filename": original_filename,
                }
        except Exception as e:
            # Rollback: eliminar archivo guardado si falla la BD
            logger.error(f"Error insertando en BD, realizando rollback de archivo: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    logger.error(f"No se pudo eliminar archivo huérfano: {file_path}")
            raise RuntimeError(f"Error al indexar imagen en base de datos: {e}")


def get_all_images() -> List[Dict]:
    """
    Devuelve la lista de todas las imágenes indexadas en la base de datos.
    """
    with SessionLocal() as session:
        rows = (
            session.query(ImageEmbedding)
            .order_by(ImageEmbedding.id.desc())
            .all()
        )
        return [
            {
                "id": row.id,
                "image_path": row.image_path,
                "original_filename": row.original_filename,
                "sha256_hash": row.sha256_hash,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]


def _resolve_physical_path(image_path: str) -> str | None:
    """
    Dado el path relativo almacenado en la BD (ej. /uploads/abc.jpg o /static/img.png),
    devuelve el path absoluto en el filesystem, o None si no lo encuentra.
    """
    if image_path.startswith("/uploads/"):
        filename = image_path.replace("/uploads/", "", 1)
        full = os.path.join(UPLOAD_FOLDER, filename)
    elif image_path.startswith("/static/"):
        filename = image_path.replace("/static/", "", 1)
        full = os.path.join("./example_images", filename)
    else:
        return None

    return full if os.path.isfile(full) else None


def delete_image_by_id(image_id: int) -> Dict:
    """
    Elimina una imagen por ID: borra el archivo físico y el registro en la BD.

    Returns:
        Dict con información del registro eliminado.

    Raises:
        ValueError: si el ID no existe.
        RuntimeError: si falla la eliminación.
    """
    with _index_lock:
        with SessionLocal() as session:
            row = session.query(ImageEmbedding).filter_by(id=image_id).first()
            if not row:
                raise ValueError(f"No existe una imagen con ID {image_id}.")

            info = {
                "id": row.id,
                "image_path": row.image_path,
                "original_filename": row.original_filename,
            }

            # Eliminar archivo físico
            physical = _resolve_physical_path(row.image_path)
            if physical:
                try:
                    os.remove(physical)
                    logger.info(f"Archivo eliminado: {physical}")
                except OSError as e:
                    logger.warning(f"No se pudo eliminar archivo {physical}: {e}")

            # Eliminar registro de la BD
            try:
                session.delete(row)
                session.commit()
                logger.info(f"Registro eliminado de la BD — ID: {image_id}")
            except Exception as e:
                session.rollback()
                raise RuntimeError(f"Error eliminando registro de la BD: {e}")

            return info


def delete_all_images() -> int:
    """
    Elimina TODAS las imágenes: archivos físicos y registros en la BD.

    Returns:
        int — cantidad de registros eliminados.
    """
    with _index_lock:
        with SessionLocal() as session:
            rows = session.query(ImageEmbedding).all()
            count = len(rows)

            # Eliminar archivos físicos
            for row in rows:
                physical = _resolve_physical_path(row.image_path)
                if physical:
                    try:
                        os.remove(physical)
                    except OSError as e:
                        logger.warning(f"No se pudo eliminar archivo {physical}: {e}")

            # Eliminar todos los registros
            try:
                session.query(ImageEmbedding).delete()
                session.commit()
                logger.info(f"Eliminados {count} registros de la BD.")
            except Exception as e:
                session.rollback()
                raise RuntimeError(f"Error eliminando registros: {e}")

            return count