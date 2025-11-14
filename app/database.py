from sqlalchemy import create_engine, Column, Integer, Text, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy import text
import logging
import os

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
    # Ajusta el tamaño del vector a lo que usa tu modelo (512 para CLIP ViT-Base)
    embedding = Column(Vector(512), nullable=False)


def create_tables():
    """
    Crea la extensión pgvector si no existe, crea las tablas y agrega índices vectoriales
    para optimizar las búsquedas de similitud.
    """
    try:
        # Crear la extensión vector (si no existe)
        with engine.connect() as conn:
            logger.info("Creando extensión pgvector...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        # Crear tablas a partir de modelos SQLAlchemy
        logger.info("Creando tablas...")
        Base.metadata.create_all(bind=engine)

        # Crear índice vectorial para búsquedas rápidas
        # HNSW es más rápido para búsquedas pero usa más memoria
        # IVFFlat es buena alternativa con menos memoria
        with engine.connect() as conn:
            # Verificar si el índice ya existe
            check_index = text("""
                SELECT 1 FROM pg_indexes 
                WHERE indexname = 'idx_embedding_hnsw'
            """)
            result = conn.execute(check_index).fetchone()
            
            if not result:
                logger.info("Creando índice vectorial HNSW para búsquedas optimizadas...")
                # HNSW con distancia L2 (euclidiana)
                # m=16 controla el número de conexiones, ef_construction=64 controla la calidad
                conn.execute(text("""
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
