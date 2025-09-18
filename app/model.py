import logging
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Inicializa el objeto, pero no carga el modelo hasta que se use por primera vez.
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ImageEmbedder inicializado en device={self.device}, modelo={model_name}")

    def _load_model(self):
        """
        Carga el modelo CLIP y el processor de Hugging Face en memoria.
        Se llama automáticamente la primera vez que se use.
        """
        if self.model is None or self.processor is None:
            try:
                logger.info(f"Cargando modelo {self.model_name}...")
                self.model = CLIPModel.from_pretrained(self.model_name)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()  # Modo inferencia
                logger.info("Modelo cargado y listo para inferencia.")
            except Exception as e:
                logger.error(f"Error al cargar el modelo {self.model_name}: {e}")
                raise RuntimeError(f"No se pudo cargar el modelo {self.model_name}") from e

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Genera el embedding para una imagen dada.
        Devuelve un vector NumPy 1-D.
        """
        # Asegurarse de que el modelo esté cargado
        self._load_model()

        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error al generar embedding: {e}")
            raise RuntimeError("Error al generar el embedding de la imagen") from e


# Instancia global (singleton) - carga perezosa
image_embedder: ImageEmbedder = ImageEmbedder()