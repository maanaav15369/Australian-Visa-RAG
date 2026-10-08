from app.embedder import Embedder
from app.vector_store import VectorStore


def test_onshore_student_holder_retrieves_current_october_2026_rule():
    query = (
        "Can a person who already holds a Student visa subclass 500 "
        "and is in Australia apply for another Student visa as a primary applicant?"
    )

    store = VectorStore()
    embedder = Embedder()

    result = store.search(
        query_embedding=embedder.embed_query(query),
        top_k=5,
        query_text=query,
    )

    versions = [
        metadata.get("version")
        for metadata in result["metadatas"][0]
    ]

    documents = [
        document.lower()
        for document in result["documents"][0]
    ]

    assert "F2026L01347" in versions
    assert any("1222(3)(aa)" in document for document in documents)


def test_defence_student_exception_retrieves_current_rule():
    query = (
        "Can a Defence student who already holds a Student visa subclass 500 "
        "and is in Australia apply for another Student visa as a primary applicant?"
    )

    store = VectorStore()
    embedder = Embedder()

    result = store.search(
        query_embedding=embedder.embed_query(query),
        top_k=5,
        query_text=query,
    )

    combined_text = " ".join(
        result["documents"][0]
    ).lower()

    assert "f2026l01347" in " ".join(
        metadata.get("version", "").lower()
        for metadata in result["metadatas"][0]
    )
    assert "defence students" in combined_text
    normalized = "".join(combined_text.split())
    assert "(3a)" in normalized
