# Module: Embedding Generation & Qdrant Database Upload

import json
import shutil
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Loading embedding model
print("Loading embedding model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer(
    "intfloat/multilingual-e5-large-instruct",
    device=device
)
print("Model loaded successfully!")

# Creating Qdrant database
db_path = "/content/qdrant_db"

if Path(db_path).exists():
    shutil.rmtree(db_path)

client = QdrantClient(path=db_path)
collection_name = "logic_corpus"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)

# Read chunks
chunk_root = Path("/content/paragraph_chunks")
txt_files = sorted(chunk_root.rglob("chunk_*.txt"))
print(f"Found {len(txt_files)} chunk files.")

texts = []
payloads = []

for txt_file in txt_files:
    text = txt_file.read_text(encoding="utf-8").strip()
    if not text:
        continue

    json_file = txt_file.with_suffix(".json")
    metadata = {}

    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    texts.append("passage: " + text)
    payloads.append({
        "text": text,
        "source_file": metadata.get("source_file", ""),
        "chunk_id": metadata.get("chunk_id", 0),
        "character_count": metadata.get(
            "character_count",
            len(text)
        ),
        "paragraph_split": metadata.get(
            "paragraph_split",
            False
        )
    })

print(f"Prepared {len(texts)} chunks for embedding.")

# Generate embeddings
print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("Embeddings generated!")

# Converting into qdrant points
points = []

for idx in range(len(payloads)):
    points.append(
        PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload=payloads[idx]
        )
    )

print(f"Prepared {len(points)} Qdrant points.")

# Uploading
UPLOAD_BATCH = 100
print("Uploading to Qdrant...")

for i in range(0, len(points), UPLOAD_BATCH):
    client.upsert(
        collection_name=collection_name,
        points=points[i:i+UPLOAD_BATCH]
    )
    print(f"Uploaded {min(i+UPLOAD_BATCH,len(points))}/{len(points)}")

print("="*60)
print("Embedding completed successfully!")
print(f"Total chunks embedded : {len(points)}")
print(f"Collection            : {collection_name}")
print("="*60)

client.close()
