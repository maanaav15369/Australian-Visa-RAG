import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.answer_generator import AnswerGenerator
from app.embedder import Embedder
from app.vector_store import VectorStore
from visa_intel.citations import FALLBACK_TEXT, extract_citations, is_insufficient
from visa_intel.conversation import (
    detect_target_jurisdictions,
    try_counsellor_intake_reply,
    try_small_talk_reply,
)
from visa_intel.contracts import (
    AVAILABLE_JURISDICTIONS,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    Jurisdiction,
    JurisdictionUnavailable,
    ensure_available,
)
from visa_intel.timing import StageTimer
from visa_intel.conversation_store import ConversationStore

logger = logging.getLogger("visa_rag.api")

_STATE = {"loaded": False}
_CONVERSATIONS = ConversationStore()


def _remember_turn(conversation_id, question, answer):
    """Store only completed turns when a conversation ID is supplied."""
    if not conversation_id:
        return

    _CONVERSATIONS.add_message(conversation_id, "user", question)
    _CONVERSATIONS.add_message(conversation_id, "assistant", answer)



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
def chat(request: ChatRequest):
    target_codes = detect_target_jurisdictions(request.question)
    selected_codes = {item.value for item in request.jurisdictions}
    conversation_id = request.conversation_id

    previous_code = (
        _CONVERSATIONS.get_jurisdiction(conversation_id)
        if conversation_id
        else None
    )

    if target_codes:
        effective_codes = target_codes
    elif previous_code:
        effective_codes = {previous_code}
    else:
        effective_codes = selected_codes

    ensure_available(request.jurisdictions)

    if request.as_of_date is not None:
        # Never silently ignore a historical request: retrieval has no as-of support yet.
        return _error(
            422,
            "invalid_request",
            "as_of_date is not supported yet; historical retrieval is not enabled.",
        )

    # Load only this conversation's recent history.
    history = (
        _CONVERSATIONS.get_history(request.conversation_id)
        if request.conversation_id
        else []
    )

    # Handle simple social messages without loading or calling the RAG stack.
    reply = try_small_talk_reply(request.question)
    if reply is not None:
        if request.conversation_id:
            _CONVERSATIONS.set_jurisdiction(
                request.conversation_id,
                next(iter(effective_codes)),
            )
        _remember_turn(request.conversation_id, request.question, reply)
        return ChatResponse(
            answer=reply,
            status="conversation",
            jurisdictions=request.jurisdictions,
            citations=[],
            timings_ms={},
            disclaimer="",
        )

    # Broad planning requests are conversational, not legal determinations.
    reply = try_counsellor_intake_reply(request.question)
    if reply is not None:
        active_codes = target_codes or effective_codes
        conversation_jurisdictions = [
            Jurisdiction(code) for code in sorted(active_codes)
        ]

        if request.conversation_id:
            _CONVERSATIONS.set_jurisdiction(
                request.conversation_id,
                next(iter(active_codes)),
            )

        _remember_turn(request.conversation_id, request.question, reply)
        return ChatResponse(
            answer=reply,
            status="conversation",
            jurisdictions=conversation_jurisdictions,
            citations=[],
            timings_ms={},
            disclaimer="",
        )

    available_codes = {item.value for item in AVAILABLE_JURISDICTIONS}
    unavailable_codes = sorted(effective_codes - available_codes)

    if unavailable_codes:
        return _error(
            422,
            "jurisdiction_unavailable",
            "No indexed evidence yet for: " + ", ".join(unavailable_codes),
        )

    if effective_codes != selected_codes:
        return _error(
            422,
            "invalid_request",
            "The conversation targets a different jurisdiction. "
            "Select it explicitly to switch destinations.",
        )

    # Load components only when the message needs immigration evidence.
    components = get_components()
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
            answer = generator.answer(
                query=request.question,
                results=results,
                history=history,
            )
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

    if request.conversation_id:
        _CONVERSATIONS.set_jurisdiction(
            request.conversation_id,
            next(iter(effective_codes)),
        )

    _remember_turn(request.conversation_id, request.question, answer)

    return ChatResponse(
        answer=answer,
        status=status_value,
        jurisdictions=request.jurisdictions,
        citations=citations,
        timings_ms=timings,
    )
