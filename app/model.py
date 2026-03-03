from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimensiones de cada componente del embedding híbrido
# ---------------------------------------------------------------------------
CLIP_DIM = 512
SPATIAL_GRID = 8          # grilla 8x8 → 64 celdas
SPATIAL_DIM = SPATIAL_GRID * SPATIAL_GRID  # 64
DHASH_SIZE = 8            # dHash de 8x8 → 64 bits
DHASH_DIM = DHASH_SIZE * DHASH_SIZE        # 64
TOTAL_DIM = CLIP_DIM + SPATIAL_DIM + DHASH_DIM  # 640

# Pesos por defecto para cada componente (deben sumar ~1.0)
DEFAULT_WEIGHTS = {
    "clip": 0.3,
    "spatial": 0.4,
    "dhash": 0.3,
}


class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Inicializa el modelo CLIP y los métodos auxiliares para generar
        embeddings híbridos que combinan:
          1. CLIP semántico (512 dims)
          2. Histograma espacial de densidad (64 dims)
          3. Perceptual difference-hash (64 dims)
        """
        logger.info(f"Cargando modelo CLIP: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.weights = DEFAULT_WEIGHTS.copy()
        logger.info(f"Modelo cargado en dispositivo: {self.device}")
        logger.info(f"Dimensión total del embedding híbrido: {TOTAL_DIM}")

    # ------------------------------------------------------------------
    # 1. Embedding semántico CLIP (existente, sin cambios funcionales)
    # ------------------------------------------------------------------
    def get_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Genera el embedding CLIP normalizado (512 dims)."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # 2. Histograma espacial de densidad de píxeles oscuros
    # ------------------------------------------------------------------
    @staticmethod
    def get_spatial_histogram(image: Image.Image, grid: int = SPATIAL_GRID,
                              threshold: int = 128) -> np.ndarray:
        """
        Divide la imagen en una grilla grid×grid y calcula la fracción
        de píxeles oscuros (< threshold en escala de grises) en cada celda.

        Esto captura *dónde* están las marcas/trazos oscuros en la imagen,
        algo que CLIP ignora por completo.

        Returns:
            numpy array de shape (grid*grid,) con valores en [0, 1].
        """
        gray = image.convert("L")
        arr = np.array(gray)
        h, w = arr.shape

        cell_h = h / grid
        cell_w = w / grid
        histogram = np.zeros(grid * grid, dtype=np.float64)

        for row in range(grid):
            for col in range(grid):
                y0 = int(row * cell_h)
                y1 = int((row + 1) * cell_h)
                x0 = int(col * cell_w)
                x1 = int((col + 1) * cell_w)
                cell = arr[y0:y1, x0:x1]
                if cell.size == 0:
                    continue
                dark_ratio = np.mean(cell < threshold)
                histogram[row * grid + col] = dark_ratio

        # Normalizar L2 para que sea comparable con los otros componentes
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
        return histogram

    # ------------------------------------------------------------------
    # 3. Perceptual difference-hash (dHash)
    # ------------------------------------------------------------------
    @staticmethod
    def get_dhash_vector(image: Image.Image,
                         hash_size: int = DHASH_SIZE) -> np.ndarray:
        """
        Calcula el dHash (difference hash) de la imagen.

        El dHash compara cada píxel con su vecino a la derecha, generando
        una huella binaria que captura gradientes locales. Es muy sensible
        a cambios sutiles en la estructura de la imagen.

        Returns:
            numpy array de shape (hash_size*hash_size,) con valores 0.0/1.0,
            normalizado L2.
        """
        # Redimensionar a (hash_size+1) × hash_size en escala de grises
        resized = image.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = np.array(resized, dtype=np.float64)

        # Comparar cada píxel con el de su derecha
        diff = pixels[:, 1:] > pixels[:, :-1]
        hash_vector = diff.flatten().astype(np.float64)

        # Normalizar L2
        norm = np.linalg.norm(hash_vector)
        if norm > 0:
            hash_vector = hash_vector / norm
        return hash_vector

    # ------------------------------------------------------------------
    # 4. Embedding combinado (el que se usa para indexar y buscar)
    # ------------------------------------------------------------------
    def get_combined_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Genera el embedding híbrido final concatenando los 3 componentes
        ponderados. El resultado es un vector de 640 dimensiones normalizado.

        Componentes y pesos por defecto:
          - CLIP (0.3): concepto semántico global
          - Spatial histogram (0.4): distribución espacial de marcas
          - dHash (0.3): estructura local / gradientes
        """
        if image.mode != "RGB":
            raise ValueError(
                f"La imagen debe estar en modo RGB, recibido: {image.mode}"
            )

        clip_emb = self.get_clip_embedding(image)       # (512,)
        spatial_emb = self.get_spatial_histogram(image)  # (64,)
        dhash_emb = self.get_dhash_vector(image)         # (64,)

        # Aplicar pesos
        clip_weighted = clip_emb * self.weights["clip"]
        spatial_weighted = spatial_emb * self.weights["spatial"]
        dhash_weighted = dhash_emb * self.weights["dhash"]

        # Concatenar
        combined = np.concatenate([clip_weighted, spatial_weighted, dhash_weighted])

        # Normalizar L2 el vector final
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        logger.debug(
            f"Embedding generado — CLIP: {clip_emb[:3]}..., "
            f"Spatial: {spatial_emb[:3]}..., dHash: {dhash_emb[:3]}..."
        )
        return combined

    # ------------------------------------------------------------------
    # Retrocompatibilidad: mantener el método original (ahora delega)
    # ------------------------------------------------------------------
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Método original mantenido por retrocompatibilidad.
        Ahora genera el embedding híbrido combinado.
        """
        return self.get_combined_embedding(image)


# Instancia global del embedder para evitar recargar el modelo en cada request
image_embedder = ImageEmbedder()
