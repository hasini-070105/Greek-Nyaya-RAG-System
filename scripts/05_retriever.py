# Module: Vector Search & Chunk Retrieval

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class QdrantRetriever:
    def __init__(self, db_path="/content/qdrant_db", collection_name="logic_corpus"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {self.device}...")
        self.embedding_model = SentenceTransformer(
            "intfloat/multilingual-e5-large-instruct", 
            device=self.device
        )
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 6):
        query_embedding = self.embedding_model.encode(
            "query: " + query, 
            normalize_embeddings=True
        ).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        ).points

        return results

    def close(self):
        self.client.close()
