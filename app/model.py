from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Inicializa el modelo CLIP para generar embeddings de imágenes.
        
        Args:
            model_name: Nombre del modelo CLIP a utilizar
        """
        logger.info(f"Cargando modelo CLIP: {model_name}")
        # Carga el modelo CLIP y su procesador
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # Modo evaluación para desactivar dropout
        logger.info(f"Modelo cargado exitosamente en dispositivo: {self.device}")

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Genera el embedding para una imagen dada.
        
        Args:
            image: Imagen PIL en formato RGB
            
        Returns:
            numpy array con el embedding de la imagen (512 dimensiones)
        """
        if image.mode != 'RGB':
            raise ValueError(f"La imagen debe estar en modo RGB, recibido: {image.mode}")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalizar el vector para consistencia
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()


# Instancia global del embedder para evitar recargar el modelo en cada request
image_embedder = ImageEmbedder()
