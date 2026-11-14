
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag import (
    index_documents,
    answer_question
)


app = FastAPI(
    title="AI Knowledge Assistant",
    description="A RAG-powered AI assistant using ChromaDB and Hugging Face.",
    version="1.0.0"
)


# -----------------------------
# REQUEST AND RESPONSE SCHEMAS
# -----------------------------

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


class IndexResponse(BaseModel):
    message: str
    indexed_documents: int


# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "application": "AI Knowledge Assistant"
    }


# -----------------------------
# INDEX DOCUMENTS
# -----------------------------

@app.post("/index", response_model=IndexResponse)
def index_knowledge():

    try:
        count = index_documents()

        return {
            "message": "Documents indexed successfully.",
            "indexed_documents": count
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# -----------------------------
# ASK QUESTIONS
# -----------------------------

@app.post("/ask", response_model=AskResponse)
def ask_question(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = answer_question(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
