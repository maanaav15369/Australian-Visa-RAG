from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.answer_generator import AnswerGenerator
from app.embedder import Embedder
from app.vector_store import VectorStore


app = FastAPI(
    title="Australian Visa RAG API",
    version="1.0.0",
    description="Version-aware Australian visa rules RAG API.",
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)


class ChatResponse(BaseModel):
    answer: str


@lru_cache(maxsize=1)
def load_components():
    return Embedder(), VectorStore(), AnswerGenerator()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "australian-visa-rag-api",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    embedder, store, generator = load_components()

    results = store.search(
        query_embedding=embedder.embed_query(request.question),
        query_text=request.question,
        top_k=5,
    )

    answer = generator.answer(
        query=request.question,
        results=results,
    )

    return ChatResponse(answer=answer)
