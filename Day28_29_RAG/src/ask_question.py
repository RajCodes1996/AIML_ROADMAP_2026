# =====================================================
# DAY 28 - RAG QUESTION ANSWERING
# Retrieve relevant documents and generate an answer
# =====================================================

from pathlib import Path

import chromadb
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Locate database

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "chroma_db"

# 2. Load embedding model

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Connect to ChromaDB

client = chromadb.PersistentClient(
    path=str(DATABASE_PATH)
)

collection = client.get_collection(
    name="ml_knowledge"
)

# 4. Load text generation model

print("Loading language model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

print("RAG assistant is ready!")

# 5. Ask questions

while True:

    question = input(
        "\nAsk a question (or type exit): "
    ).strip()

    if question.lower() == "exit":
        print("Assistant closed.")
        break

    if not question:
        print("Please enter a question.")
        continue

    # 6. Convert question into embedding

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # 7. Retrieve relevant documents

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(3, collection.count())
    )

    retrieved_chunks = results["documents"][0]

    context = "\n\n".join(retrieved_chunks)

    # 8. Build prompt

    prompt = f"""
Answer the question using only the information
provided in the context.

If the answer is not available in the context,
say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
"""

    # 9. Generate answer

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 10. Display answer and retrieved context

    print("\nRetrieved Documents:")
    for chunk in retrieved_chunks:
        print("-", chunk)

    print("\nAI Answer:")
    print(answer)