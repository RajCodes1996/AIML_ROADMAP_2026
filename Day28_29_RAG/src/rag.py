# =====================================================
# RAG CORE LOGIC (src/rag.py)
# =====================================================

from pathlib import Path

import torch
import chromadb

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# -----------------------------
# PROJECT PATHS AND CONFIGURATION
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "knowledge.txt"
DATABASE_PATH = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"

COLLECTION_NAME = "ml_knowledge"


# -----------------------------
# LAZY MODEL LOADING
# -----------------------------

embedding_model = None
tokenizer = None
model = None
collection = None


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        print("Loading embedding model...")

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return embedding_model


def get_generation_model():
    global tokenizer, model

    if model is None or tokenizer is None:
        print("Loading language model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME).to(device)

    return tokenizer, model


# -----------------------------
# CHROMADB CONNECTION
# -----------------------------

def get_collection():
    global collection

    if collection is None:
        client = chromadb.PersistentClient(
            path=str(DATABASE_PATH)
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    return collection


def get_collection_count():
    return get_collection().count()


# -----------------------------
# DOCUMENT INDEXING
# -----------------------------

def index_documents():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {DATA_PATH}"
        )

    text = DATA_PATH.read_text(encoding="utf-8")

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    if not chunks:
        raise ValueError("No documents found to index.")

    model = get_embedding_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    db = get_collection()

    # Remove previous records before rebuilding the index
    existing_ids = db.get()["ids"]

    if existing_ids:
        db.delete(ids=existing_ids)

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    db.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return len(chunks)


# -----------------------------
# QUESTION ANSWERING WITH RAG
# -----------------------------

def answer_question(question: str):

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    db = get_collection()

    document_count = db.count()

    if document_count == 0:
        raise ValueError(
            "Knowledge base is empty. Please index documents first."
        )

    model = get_embedding_model()

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    results = db.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=min(3, document_count)
    )

    retrieved_documents = results["documents"][0]

    context = "\n".join(retrieved_documents)

    prompt = f"""
Answer the question using only the context below.

If the answer is not available in the context,
say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
"""

    tokenizer, gen_model = get_generation_model()
    device = next(gen_model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "sources": retrieved_documents
    }