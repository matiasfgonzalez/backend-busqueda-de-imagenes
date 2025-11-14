# Backend - Búsqueda de Imágenes con IA

Sistema de búsqueda de imágenes similares utilizando embeddings generados con CLIP (Contrastive Language-Image Pre-training) y búsqueda vectorial con pgvector.

## 🚀 Características

- **Embeddings con CLIP**: Utiliza el modelo `openai/clip-vit-base-patch32` para generar representaciones vectoriales de imágenes
- **Búsqueda vectorial rápida**: PostgreSQL con extensión pgvector e índices HNSW para búsquedas optimizadas
- **API REST con FastAPI**: Endpoints modernos y documentados automáticamente
- **Health checks**: Monitoreo del estado del servicio y la base de datos
- **Logging estructurado**: Trazabilidad completa de operaciones
- **Validaciones**: Verificación de dimensiones de vectores y tipos de archivos

## 📋 Requisitos

- Python 3.10+
- PostgreSQL con extensión pgvector
- Docker y Docker Compose (opcional)

## 🛠️ Instalación

### Con Docker (Recomendado)

```bash
# Desde la raíz del proyecto
docker-compose up --build
```

### Manual

1. Instalar dependencias:

```bash
cd backend
pip install -r requirements.txt
```

2. Configurar variables de entorno:

```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

3. Ejecutar la aplicación:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📁 Estructura del Proyecto

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # Aplicación FastAPI y endpoints
│   ├── model.py         # Modelo CLIP para embeddings
│   ├── database.py      # Configuración de base de datos
│   └── utils.py         # Funciones auxiliares
├── example_images/      # Imágenes de ejemplo
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## 🔌 API Endpoints

### Health Check

```
GET /health
```

Verifica el estado del servicio y la conexión a la base de datos.

**Respuesta:**

```json
{
  "status": "healthy",
  "service": "image-search-backend",
  "database": "connected"
}
```

### Búsqueda de Imágenes Similares

```
POST /search-similar-images/
```

Busca imágenes similares a la imagen subida.

**Parámetros:**

- `file`: Archivo de imagen (multipart/form-data)

**Respuesta:**

```json
{
  "results": [
    {
      "id": 1,
      "similarity": 0.95,
      "path": "/static/imagen1.jpg"
    },
    {
      "id": 2,
      "similarity": 0.87,
      "path": "/static/imagen2.jpg"
    }
  ]
}
```

### Archivos Estáticos

```
GET /static/{filename}
```

Sirve las imágenes almacenadas.

## ⚙️ Variables de Entorno

| Variable               | Descripción                                        | Default                                                      |
| ---------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| `DATABASE_URL`         | URL de conexión a PostgreSQL                       | `postgresql://postgres:postgres@localhost:5432/image_search` |
| `ALLOWED_ORIGINS`      | Orígenes permitidos para CORS (separados por coma) | `http://localhost:3000`                                      |
| `SIMILARITY_THRESHOLD` | Umbral mínimo de similitud (0.0-1.0)               | `0.2`                                                        |
| `LOG_LEVEL`            | Nivel de logging                                   | `INFO`                                                       |

## 🗄️ Base de Datos

### Extensión pgvector

El sistema requiere la extensión pgvector de PostgreSQL para almacenar y buscar vectores eficientemente.

### Índice Vectorial

Se crea automáticamente un índice HNSW (Hierarchical Navigable Small World) para optimizar las búsquedas:

```sql
CREATE INDEX idx_embedding_hnsw ON image_embeddings
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);
```

### Esquema

```sql
CREATE TABLE image_embeddings (
    id SERIAL PRIMARY KEY,
    image_path TEXT NOT NULL UNIQUE,
    embedding VECTOR(512) NOT NULL
);
```

## 🔧 Optimizaciones Implementadas

1. **Context Managers**: Gestión automática de sesiones de base de datos
2. **Batch Processing**: Inserción de embeddings en lote para mejor performance
3. **Normalización de Vectores**: Los embeddings se normalizan para consistencia
4. **Índices Vectoriales**: HNSW para búsquedas O(log n) en lugar de O(n)
5. **Connection Pooling**: Reutilización de conexiones a la base de datos
6. **Modelo en Modo Eval**: Desactivación de dropout para inferencia consistente

## 📊 Performance

- **Búsqueda**: ~10-50ms para bases de datos de hasta 10,000 imágenes (con índice HNSW)
- **Generación de Embedding**: ~100-200ms por imagen (CPU), ~20-50ms (GPU)
- **Carga Inicial**: Procesamiento de ~10 imágenes/segundo

## 🧪 Testing

```bash
# Ejecutar tests
pytest

# Con cobertura
pytest --cov=app tests/
```

## 📝 Logging

Los logs incluyen:

- Inicialización del modelo y base de datos
- Procesamiento de imágenes
- Errores y excepciones con stack traces
- Métricas de búsqueda

## 🐛 Troubleshooting

### Error: "operator does not exist: vector <-> numeric[]"

**Solución**: El vector debe convertirse explícitamente usando `::vector` en la query SQL. Ya implementado en `utils.py`.

### Error: "No se pueden cargar imágenes"

**Solución**: Verificar que la carpeta `example_images` existe y contiene imágenes válidas (.jpg, .png, .jpeg, etc.)

### Performance lenta en búsquedas

**Solución**:

1. Verificar que el índice HNSW está creado
2. Aumentar `m` y `ef_construction` en el índice
3. Considerar usar GPU para embeddings

## 📚 Documentación API

Una vez iniciado el servidor, visita:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT.
