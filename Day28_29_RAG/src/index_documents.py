
# =====================================================
# DAY 28 - DOCUMENT INDEXING
# Create embeddings and store them in ChromaDB
# =====================================================

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# 1. Locate project folders

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "knowledge.txt"

DATABASE_PATH = PROJECT_ROOT / "chroma_db"

# 2. Load embedding model

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Read document

text = DOCUMENT_PATH.read_text(encoding="utf-8")

# 4. Split document into chunks

chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]

print("Total document chunks:", len(chunks))

# 5. Generate embeddings

print("Generating embeddings...")

embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
).tolist()

# 6. Connect to persistent ChromaDB

client = chromadb.PersistentClient(
    path=str(DATABASE_PATH)
)

# Create or reuse collection
collection = client.get_or_create_collection(
    name="ml_knowledge"
)

# 7. Store documents and embeddings

# Replace existing entries so rerunning this script
# does not create duplicate records.

existing = collection.get()

if existing["ids"]:
    collection.delete(ids=existing["ids"])

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("\nIndexing completed successfully!")
print("Stored chunks:", collection.count())