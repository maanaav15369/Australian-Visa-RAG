import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.answer_generator import AnswerGenerator
from app.embedder import Embedder
from app.vector_store import VectorStore
from visa_intel.citations import FALLBACK_TEXT, extract_citations, is_insufficient
from visa_intel.contracts import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    JurisdictionUnavailable,
    ensure_available,
)
from visa_intel.timing import StageTimer

logger = logging.getLogger("visa_rag.api")

_STATE = {"loaded": False}


class ComponentsUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def load_components():
    components = (Embedder(), VectorStore(), AnswerGenerator())
    _STATE["loaded"] = True
    return components


def get_components():
    try:
        return load_components()
    except Exception as exc:
        logger.exception("Could not initialise RAG components")
        raise ComponentsUnavailable from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup so the first /chat does not pay the cold start.
    # Set VISA_RAG_WARMUP=0 to skip. A failure is logged, not fatal.
    if os.getenv("VISA_RAG_WARMUP", "1") != "0":
        try:
            load_components()
            logger.info("RAG components loaded at startup.")
        except Exception:
            logger.exception("Warm-up failed; /chat will retry on first request.")
    yield


app = FastAPI(
    title="Australian Visa RAG API",
    version="1.1.0",
    description="Version-aware visa rules RAG API (Australia indexed).",
    lifespan=lifespan,
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(code=code, message=message).model_dump()
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(RequestValidationError)
async def _invalid_request(request, exc):
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"] if p != "body")
        parts.append(f"{loc}: {err['msg']}" if loc else str(err["msg"]))
    return _error(422, "invalid_request", "; ".join(parts) or "Invalid request.")


@app.exception_handler(JurisdictionUnavailable)
async def _jurisdiction_unavailable(request, exc):
    return _error(422, "jurisdiction_unavailable", str(exc))


@app.exception_handler(ComponentsUnavailable)
async def _components_unavailable(request, exc):
    return _error(503, "internal_error", "Answer components are not available.")


def _classify_upstream(exc: Exception):
    status = getattr(exc, "status_code", None)
    if status == 429:
        return 429, "rate_limited", "The answer service is rate-limited. Retry shortly."
    if status in (408, 504) or "timeout" in type(exc).__name__.lower():
        return 504, "timeout", "The answer service timed out. Please retry."
    return 502, "upstream_error", "The answer service failed. Please retry."


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "australian-visa-rag-api",
        "components_loaded": _STATE["loaded"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, components=Depends(get_components)):
    ensure_available(request.jurisdictions)

    if request.as_of_date is not None:
        # Never silently ignore a historical request: retrieval has no as-of support yet.
        return _error(
            422,
            "invalid_request",
            "as_of_date is not supported yet; historical retrieval is not enabled.",
        )

    embedder, store, generator = components
    # Only AU is available, so exactly one jurisdiction reaches this point.
    jurisdiction = request.jurisdictions[0]
    timer = StageTimer()

    with timer.stage("embed_query"):
        query_embedding = embedder.embed_query(request.question)

    with timer.stage("retrieve"):
        results = store.search(
            query_embedding=query_embedding,
            query_text=request.question,
            top_k=5,
        )

    try:
        with timer.stage("generate"):
            answer = generator.answer(query=request.question, results=results)
    except Exception as exc:
        logger.warning("Answer generation failed: %s", type(exc).__name__)
        status, code, message = _classify_upstream(exc)
        return _error(status, code, message)

    citations = extract_citations(answer, results, jurisdiction)

    if is_insufficient(answer):
        status_value = "insufficient_evidence"
    elif not citations:
        logger.warning("Answer had no valid source markers; returning fallback.")
        answer, status_value = FALLBACK_TEXT, "insufficient_evidence"
    else:
        status_value = "answered"

    timings = timer.as_dict()
    timings["stages_total"] = timer.total_ms()

    return ChatResponse(
        answer=answer,
        status=status_value,
        jurisdictions=request.jurisdictions,
        citations=citations,
        timings_ms=timings,
    )
