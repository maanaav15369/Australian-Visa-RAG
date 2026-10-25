from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from visa_intel.contracts import (
    ChatRequest,
    ChatResponse,
    Citation,
    Jurisdiction,
    JurisdictionUnavailable,
    StreamEventType,
    ensure_available,
    format_sse,
)
from visa_intel.timing import StageTimer


def test_defaults_to_australia():
    assert ChatRequest(question="What is a Student visa?").jurisdictions == [Jurisdiction.AU]


def test_single_jurisdiction_is_normalised():
    req = ChatRequest(question="Student visa rules?", jurisdiction="AU")
    assert req.jurisdictions == [Jurisdiction.AU]


def test_both_jurisdiction_fields_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(question="Student visa?", jurisdiction="AU", jurisdictions=["AU"])


def test_duplicate_jurisdictions_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(question="Compare visas", jurisdictions=["AU", "AU"])


def test_unknown_country_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(question="Visa rules?", jurisdictions=["XX"])


def test_blank_question_rejected_and_whitespace_stripped():
    with pytest.raises(ValidationError):
        ChatRequest(question="   ")
    assert ChatRequest(question="  hello visa  ").question == "hello visa"


def test_future_as_of_date_rejected_past_accepted():
    with pytest.raises(ValidationError):
        ChatRequest(question="Rules?", as_of_date=date.today() + timedelta(days=2))
    req = ChatRequest(question="Rules?", as_of_date=date(2024, 1, 1))
    assert req.is_historical is True
    assert ChatRequest(question="Rules?").is_historical is False


def test_bad_conversation_id_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(question="Rules?", conversation_id="a b/c")
    assert ChatRequest(question="Rules?", conversation_id="conv_12345678").conversation_id


def test_unknown_request_field_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(question="Rules?", surprise="x")


def test_unindexed_jurisdiction_is_reported_not_substituted():
    ensure_available([Jurisdiction.AU])
    with pytest.raises(JurisdictionUnavailable) as exc:
        ensure_available([Jurisdiction.AU, Jurisdiction.CA])
    assert exc.value.missing == [Jurisdiction.CA]


def test_answered_response_requires_citation():
    with pytest.raises(ValidationError):
        ChatResponse(answer="x", status="answered", jurisdictions=["AU"])
    cite = Citation(marker="S1", chunk_id="c1", jurisdiction="AU")
    assert ChatResponse(
        answer="x [S1]", status="answered", jurisdictions=["AU"], citations=[cite]
    ).status == "answered"


def test_insufficient_evidence_needs_no_citation():
    resp = ChatResponse(
        answer="Insufficient authoritative evidence found.",
        status="insufficient_evidence",
        jurisdictions=["AU"],
    )
    assert resp.citations == []
    assert "not legal advice" in resp.disclaimer


def test_sse_format():
    assert (
        format_sse(StreamEventType.TOKEN, {"text": "hi"})
        == 'event: token\ndata: {"text":"hi"}\n\n'
    )


def test_stage_timer_accumulates_and_survives_exceptions():
    import time

    timer = StageTimer()
    with timer.stage("a"):
        time.sleep(0.01)
    with pytest.raises(RuntimeError):
        with timer.stage("b"):
            raise RuntimeError("boom")
    stages = timer.as_dict()
    assert stages["a"] >= 5
    assert "b" in stages
    assert timer.total_ms() >= stages["a"]
