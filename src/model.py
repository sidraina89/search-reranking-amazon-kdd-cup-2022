from abc import ABC, abstractmethod
from typing import Union, List
from sentence_transformers import SentenceTransformer,util
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class CosineSimilarityModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def forward(self, sentences: List[str]) -> np.array:
        return self.model.encode(
            sentences=sentences,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    def predict(self, queries: np.array, documents: np.array) -> np.array:
        return util.pairwise_dot_score(np.vstack(queries), np.vstack(documents))
