# Module: Semantic Retrieval Testing with Qdrant

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Loading embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "intfloat/multilingual-e5-large-instruct",
    device=device
)

# Connecting to Qdrant
client = QdrantClient(path="/content/qdrant_db")
collection_name = "logic_corpus"

# Query
query = input("Enter your question: ")

# Query embedding
query_embedding = model.encode(
    "query: " + query,
    normalize_embeddings=True
).tolist()

# Search Qdrant
results = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=6
).points

# Print retrieved chunks
print("\n")
print("="*80)
print("TOP RETRIEVED CHUNKS")
print("="*80)

for i, point in enumerate(results, start=1):
    print(f"\nRank : {i}")
    print("-"*80)
    print("Similarity Score :", round(point.score, 4))
    print("Source File :", point.payload.get("source_file", ""))
    print("Chunk ID :", point.payload.get("chunk_id", ""))
    print("Character Count :", point.payload.get("character_count", ""))
    print("\nRetrieved Text:\n")
    print(point.payload.get("text", ""))
    print("\n" + "="*80)

client.close()
