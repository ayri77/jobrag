from sentence_transformers import SentenceTransformer
import numpy as np

from jobrag.settings import SETTINGS

class EmbeddingModel:
    def __init__(self, model_name: str = SETTINGS.EMBED_MODEL):
        self.model_name = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model_name.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embeddings