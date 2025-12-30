import faiss
import numpy as np
import json
from app.config import INDEX_DIR

class VectorStore:
    def __init__(self, dim: int):
        self.index_path = INDEX_DIR / "faiss.index"
        self.meta_path = INDEX_DIR / "metadata.json"
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.meta_path.read_text())

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        self.index.add(vectors)
        self.metadata.extend(metadatas)
        self._persist()

    def search(self, query_vector: np.ndarray, top_k: int):
        D, I = self.index.search(
            np.array([query_vector]), top_k
        )
        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def _persist(self):
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, indent=2))
