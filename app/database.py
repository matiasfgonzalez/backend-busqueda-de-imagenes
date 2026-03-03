from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy import text
from datetime import datetime, timezone
import logging
import os

from .model import TOTAL_DIM

logger = logging.getLogger(__name__)

# Obtener DATABASE_URL de env (fallback para desarrollo local)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/image_search")

# engine y session
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(Text, nullable=False, unique=True)
    # Dimensión dinámica: 640 para embedding híbrido (CLIP 512 + spatial 64 + dHash 64)
    embedding = Column(Vector(TOTAL_DIM), nullable=False)
    sha256_hash = Column(Text, nullable=True, unique=True, index=True)
    original_filename = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=True, default=lambda: datetime.now(timezone.utc))


def create_tables():
    """
    Crea la extensión pgvector si no existe, crea las tablas y agrega índices vectoriales
    para optimizar las búsquedas de similitud.

    Si la tabla ya existe con una dimensión de embedding diferente (ej: 512),
    se realiza una migración automática: se eliminan los datos antiguos, se
    recrea la columna con la nueva dimensión, y se vuelve a crear el índice.
    """
    try:
        # Crear la extensión vector (si no existe)
        with engine.connect() as conn:
            logger.info("Creando extensión pgvector...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        # ------------------------------------------------------------------
        # Migración: detectar si la tabla existe con dimensión diferente
        # ------------------------------------------------------------------
        with engine.connect() as conn:
            check_table = text("""
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'image_embeddings'
            """)
            table_exists = conn.execute(check_table).fetchone()

            if table_exists:
                # Verificar la dimensión actual del vector
                check_dim = text("""
                    SELECT atttypmod FROM pg_attribute
                    JOIN pg_class ON pg_attribute.attrelid = pg_class.oid
                    WHERE pg_class.relname = 'image_embeddings'
                      AND pg_attribute.attname = 'embedding'
                """)
                dim_result = conn.execute(check_dim).fetchone()

                if dim_result and dim_result[0] != TOTAL_DIM:
                    old_dim = dim_result[0]
                    logger.warning(
                        f"Dimensión de embedding cambió de {old_dim} a {TOTAL_DIM}. "
                        f"Migrando tabla (se eliminarán embeddings existentes)..."
                    )
                    # Eliminar índice HNSW si existe
                    conn.execute(text(
                        "DROP INDEX IF EXISTS idx_embedding_hnsw;"
                    ))
                    # Eliminar todos los registros (los embeddings viejos son incompatibles)
                    conn.execute(text("DELETE FROM image_embeddings;"))
                    # Alterar la columna al nuevo tamaño
                    conn.execute(text(
                        f"ALTER TABLE image_embeddings "
                        f"ALTER COLUMN embedding TYPE vector({TOTAL_DIM});"
                    ))
                    conn.commit()
                    logger.info(f"Migración completada: columna embedding ahora es vector({TOTAL_DIM})")

        # Crear tablas a partir de modelos SQLAlchemy
        logger.info("Creando tablas...")
        Base.metadata.create_all(bind=engine)

        # Migración: agregar columnas nuevas si no existen (para tablas ya creadas)
        with engine.connect() as conn:
            for col_name, col_def in [
                ("sha256_hash", "TEXT UNIQUE"),
                ("original_filename", "TEXT"),
                ("created_at", "TIMESTAMP"),
            ]:
                check = text("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'image_embeddings' AND column_name = :col
                """)
                exists = conn.execute(check, {"col": col_name}).fetchone()
                if not exists:
                    logger.info(f"Agregando columna '{col_name}' a image_embeddings...")
                    conn.execute(text(
                        f"ALTER TABLE image_embeddings ADD COLUMN {col_name} {col_def};"
                    ))
                    conn.commit()

        # Crear índice vectorial para búsquedas rápidas
        with engine.connect() as conn:
            check_index = text("""
                SELECT 1 FROM pg_indexes 
                WHERE indexname = 'idx_embedding_hnsw'
            """)
            result = conn.execute(check_index).fetchone()
            
            if not result:
                logger.info("Creando índice vectorial HNSW para búsquedas optimizadas...")
                conn.execute(text(f"""
                    CREATE INDEX idx_embedding_hnsw ON image_embeddings 
                    USING hnsw (embedding vector_l2_ops)
                    WITH (m = 16, ef_construction = 64);
                """))
                conn.commit()
                logger.info("Índice vectorial creado exitosamente")
            else:
                logger.info("Índice vectorial ya existe")
                
        logger.info("Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"Error al crear tablas o índices: {e}")
        raise
