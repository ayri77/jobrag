from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import faiss

@dataclass
class FaissStore:
    dim: int
    index: faiss.Index

    @classmethod
    def new_cosine(cls, dim: int) -> "FaissStore":
        # we use inner product on normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(dim)
        return cls(dim=dim, index=index)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        scores, idx = self.index.search(query_vec, top_k)
        return scores[0], idx[0]


