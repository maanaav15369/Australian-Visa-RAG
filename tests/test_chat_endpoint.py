import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from visa_intel.citations import FALLBACK_TEXT

RESULTS = {
    "ids": [["c1", "c2"]],
    "documents": [["text one", "text two"]],
    "metadatas": [[
        {"title": "Test Doc", "status": "CURRENT", "version": "v1",
         "line_start": 10, "line_end": 20, "source_url": "https://example.org/a"},
        {"title": "Other Doc"},
    ]],
}


class FakeEmbedder:
    def embed_query(self, text):
        return [0.0]


class FakeStore:
    def __init__(self):
        self.calls = 0

    def search(self, query_embedding, query_text, top_k):
        self.calls += 1
        return RESULTS


class FakeGenerator:
    def __init__(self, text, error):
        self.text, self.error = text, error

    def answer(self, query, results):
        if self.error:
            raise self.error
        return self.text


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__("secret-detail")
        self.status_code = status_code


class FakeTimeoutError(Exception):
    pass


@pytest.fixture
def make_client():
    def _make(text="Test answer. [S1]", error=None):
        store = FakeStore()
        components = (FakeEmbedder(), store, FakeGenerator(text, error))
        api_main.app.dependency_overrides[api_main.get_components] = lambda: components
        return TestClient(api_main.app), store

    yield _make
    api_main.app.dependency_overrides.clear()


def test_old_style_request_still_works(make_client):
    client, _ = make_client()
    resp = client.post("/chat", json={"question": "What is a Student visa?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Test answer. [S1]"
    assert body["status"] == "answered"
    assert body["jurisdictions"] == ["AU"]
    cite = body["citations"][0]
    assert (cite["marker"], cite["chunk_id"], cite["provision"]) == ("S1", "c1", "lines 10-20")
    assert {"embed_query", "retrieve", "generate", "stages_total"} <= set(body["timings_ms"])
    assert "not legal advice" in body["disclaimer"]


def test_answer_without_valid_markers_falls_back(make_client):
    client, _ = make_client(text="Confident but uncited. [S9]")
    body = client.post("/chat", json={"question": "Some question"}).json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"] == FALLBACK_TEXT
    assert body["citations"] == []


def test_model_fallback_is_reported(make_client):
    client, _ = make_client(text=FALLBACK_TEXT)
    body = client.post("/chat", json={"question": "Some question"}).json()
    assert body["status"] == "insufficient_evidence"


def test_unavailable_country_is_rejected_before_retrieval(make_client):
    client, store = make_client()
    resp = client.post("/chat", json={"question": "Work visa?", "jurisdiction": "CA"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "jurisdiction_unavailable"
    assert store.calls == 0


def test_as_of_date_is_rejected_not_ignored(make_client):
    client, store = make_client()
    resp = client.post("/chat", json={"question": "Rules?", "as_of_date": "2024-01-01"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"
    assert store.calls == 0


def test_blank_question_gives_typed_error(make_client):
    client, _ = make_client()
    resp = client.post("/chat", json={"question": "   "})
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"


def test_rate_limit_maps_to_429_without_leaking_details(make_client):
    client, _ = make_client(error=FakeHTTPError(429))
    resp = client.post("/chat", json={"question": "Some question"})
    assert resp.status_code == 429
    assert resp.json()["code"] == "rate_limited"
    assert "secret-detail" not in resp.text


def test_timeout_maps_to_504(make_client):
    client, _ = make_client(error=FakeTimeoutError("slow"))
    resp = client.post("/chat", json={"question": "Some question"})
    assert (resp.status_code, resp.json()["code"]) == (504, "timeout")


def test_other_upstream_failure_maps_to_502(make_client):
    client, _ = make_client(error=RuntimeError("secret-detail"))
    resp = client.post("/chat", json={"question": "Some question"})
    assert (resp.status_code, resp.json()["code"]) == (502, "upstream_error")
    assert "secret-detail" not in resp.text


def test_components_unavailable_gives_503(monkeypatch):
    def boom():
        raise RuntimeError("COHERE_API_KEY is not configured.")

    monkeypatch.setattr(api_main, "load_components", boom)
    resp = TestClient(api_main.app).post("/chat", json={"question": "Some question"})
    assert (resp.status_code, resp.json()["code"]) == (503, "internal_error")
    assert "COHERE" not in resp.text


def test_health_keeps_old_fields():
    body = TestClient(api_main.app).get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "australian-visa-rag-api"
    assert "components_loaded" in body


def test_startup_loads_components(monkeypatch):
    calls = []
    monkeypatch.setattr(api_main, "load_components", lambda: calls.append(1))
    with TestClient(api_main.app):
        pass
    assert calls == [1]


def test_failed_warmup_does_not_stop_the_server(monkeypatch):
    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(api_main, "load_components", boom)
    with TestClient(api_main.app) as client:
        assert client.get("/health").status_code == 200
