from fastapi.testclient import TestClient

import api.main as api_main
from visa_intel.conversation import (
    detect_target_jurisdictions,
    try_counsellor_intake_reply,
    try_small_talk_reply,
)


def test_greeting_is_fast_and_deterministic():
    assert try_small_talk_reply("Hii") == (
        "Hi! How can I help with your immigration plans today?"
    )


def test_broad_australian_study_planning_asks_one_question():
    reply = try_counsellor_intake_reply(
        "I am thinking about studying in Australia, but I do not know "
        "where to start. Can you guide me?"
    )
    assert reply is not None
    assert "what level of study are you considering" in reply.lower()
    assert reply.count("?") == 1


def test_specific_requirements_question_uses_rag():
    assert try_counsellor_intake_reply(
        "I want to study in Australia. What are the Student visa requirements?"
    ) is None


def test_specific_fee_question_uses_rag():
    assert try_counsellor_intake_reply(
        "What is the current base application charge for Australian Subclass 500?"
    ) is None


def test_canadian_citizenship_does_not_override_australian_destination():
    assert detect_target_jurisdictions(
        "Can a Canadian citizen apply for an Australian Subclass 500 visa?"
    ) == {"AU"}


def test_canadian_study_permit_detects_canada():
    assert "CA" in detect_target_jurisdictions(
        "What are the requirements for a Canadian study permit?"
    )


def test_canadian_legal_question_is_rejected_before_rag(monkeypatch):
    def unexpected_load():
        raise AssertionError("Unsupported country must not load RAG components.")

    monkeypatch.setattr(api_main, "get_components", unexpected_load)
    response = TestClient(api_main.app).post(
        "/chat",
        json={"question": "What are the requirements for a Canadian study permit?"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "jurisdiction_unavailable"
    assert "CA" in response.json()["message"]



def test_uk_planning_response_reports_gb_not_default_au(monkeypatch):
    def unexpected_load():
        pytest.fail("Planning response must bypass RAG component loading.")

    monkeypatch.setattr(api_main, "get_components", unexpected_load)
    response = TestClient(api_main.app).post(
        "/chat",
        json={"question": "I am thinking about studying in the UK. Where do I start?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "conversation"
    assert body["jurisdictions"] == ["GB"]
    assert body["citations"] == []
    assert "does not yet" in body["answer"]
