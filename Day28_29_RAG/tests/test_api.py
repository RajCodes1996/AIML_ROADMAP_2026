
from fastapi.testclient import TestClient

import src.app as app_module

client = TestClient(app_module.app)

# -----------------------------
# TEST HEALTH ENDPOINT
# -----------------------------

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"

# -----------------------------
# TEST INDEX ENDPOINT
# -----------------------------

def test_index(monkeypatch):

    def fake_index_documents():
        return 5

    monkeypatch.setattr(
        app_module,
        "index_documents",
        fake_index_documents
    )

    response = client.post("/index")

    assert response.status_code == 200

    assert response.json()["indexed_documents"] == 5

# -----------------------------
# TEST ASK ENDPOINT
# -----------------------------

def test_ask(monkeypatch):

    def fake_answer_question(question):

        return {
            "answer": "Deep learning uses neural networks.",
            "sources": [
                "Deep learning is a subset of machine learning."
            ]
        }

    monkeypatch.setattr(
        app_module,
        "answer_question",
        fake_answer_question
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is deep learning?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is deep learning?"

    assert "answer" in data

    assert len(data["sources"]) > 0

# -----------------------------
# TEST INVALID QUESTION
# -----------------------------

def test_empty_question():

    response = client.post(
        "/ask",
        json={
            "question": ""
        }
    )

    assert response.status_code == 422