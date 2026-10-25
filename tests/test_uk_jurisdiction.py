import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from visa_intel.contracts import (
    AVAILABLE_JURISDICTIONS,
    Jurisdiction,
    JurisdictionUnavailable,
    ensure_available,
)
from visa_intel.conversation import (
    detect_target_jurisdictions,
    try_counsellor_intake_reply,
)


def test_gb_is_registered_but_not_enabled():
    assert Jurisdiction("GB") is Jurisdiction.GB
    assert Jurisdiction.GB not in AVAILABLE_JURISDICTIONS
    with pytest.raises(JurisdictionUnavailable):
        ensure_available([Jurisdiction.GB])


def test_uk_student_visa_is_detected_as_gb():
    assert detect_target_jurisdictions(
        "What are the requirements for a UK student visa?"
    ) == {"GB"}


def test_united_kingdom_destination_is_detected():
    assert detect_target_jurisdictions(
        "I am thinking about studying in the United Kingdom."
    ) == {"GB"}


def test_british_nationality_does_not_override_australian_destination():
    assert detect_target_jurisdictions(
        "Can a British citizen apply for a visa in Australia?"
    ) == {"AU"}


def test_uk_planning_response_is_honest_about_support_status():
    reply = try_counsellor_intake_reply(
        "I am thinking about studying in the UK. Where do I start?"
    )
    assert reply is not None
    assert "United Kingdom" in reply
    assert "does not yet" in reply


def test_uk_legal_question_is_rejected_before_retrieval(monkeypatch):
    def unexpected_load():
        pytest.fail("UK is not indexed; RAG components must not be loaded.")

    monkeypatch.setattr(api_main, "get_components", unexpected_load)
    response = TestClient(api_main.app).post(
        "/chat",
        json={
            "question": "What are the requirements for a UK student visa?"
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "jurisdiction_unavailable"
    assert "GB" in response.json()["message"]
