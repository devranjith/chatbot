# ingest.py - Run this whenever you update rsd_data.md
import os
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load embedding model (runs first time download ~90MB)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2') # 384 dimension, fast

# 2. Create Chroma client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rsd_knowledge")

# 3. Read your knowledge file and split into chunks
with open("knowledge/rsd_data.md", "r", encoding="utf-8") as f:
    text = f.read()

# Split by lines that start with "-"
chunks = [line.strip() for line in text.split("\n") if line.strip().startswith("-")]
print(f"Found {len(chunks)} chunks")

# 4. Clear old data
if collection.count() > 0:
    # delete all old
    ids = collection.get()['ids']
    collection.delete(ids=ids)

# 5. Create embeddings and store
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )
    print(f"Added: {chunk[:60]}...")

print(f"\nDone! Stored {collection.count()} chunks in./chroma_db")