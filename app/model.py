from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        # Carga el modelo CLIP y su procesador
        # CLIP es excelente para generar embeddings de imagen que capturan el contenido visual.
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Genera el embedding para una imagen dada.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy().flatten()

# Instancia global del embedder para evitar recargar el modelo en cada request
image_embedder = ImageEmbedder()