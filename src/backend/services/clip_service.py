import logging

import torch
from transformers import CLIPModel, CLIPProcessor

from ..core.config import settings

logger = logging.getLogger(__name__)


class CLIPService:
    def __init__(self):
        self.device = settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU instead")
            self.device = "cpu"

        self.model_name = settings.clip_model
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load CLIP model and processor"""
        logger.info(f"Loading CLIP model: {self.model_name}")
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            logger.info(f"CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}")
            raise

    def encode_text(self, texts) -> torch.Tensor:
        """Encode text to embedding"""
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            
            # Handle both tensor and ModelOutput types
            if isinstance(text_features, torch.Tensor):
                embeddings = text_features
            elif hasattr(text_features, 'pooler_output'):
                embeddings = text_features.pooler_output
            elif hasattr(text_features, 'last_hidden_state'):
                # If we get last_hidden_state (3D), take the pooled token (first token)
                embeddings = text_features.last_hidden_state[:, 0, :]
            else:
                embeddings = torch.tensor(text_features)
            
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu()

    def encode_image(self, image) -> torch.Tensor:
        """Encode image to embedding"""
        image_inputs = self.processor(images=image, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            
            # Handle both tensor and ModelOutput types
            if isinstance(image_features, torch.Tensor):
                embeddings = image_features
            elif hasattr(image_features, 'pooler_output'):
                embeddings = image_features.pooler_output
            elif hasattr(image_features, 'last_hidden_state'):
                # If we get last_hidden_state (3D), take the pooled token (first token)
                embeddings = image_features.last_hidden_state[:, 0, :]
            else:
                embeddings = torch.tensor(image_features)
            
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu()


clip_service = CLIPService()
