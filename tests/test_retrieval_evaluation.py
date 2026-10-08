import re

from app.embedder import Embedder
from app.vector_store import VectorStore


CASES = [
    {
        "name": "current student visa requirements",
        "query": "What are the eligibility requirements for the Student visa subclass 500?",
        "required_versions": {"F2026L01347", "C288"},
        "required_phrases": ["subclass 500", "primary criteria"],
    },
    {
        "name": "current student visa fee",
        "query": "What is the current base application charge for the Student visa subclass 500?",
        "required_versions": {"F2026L00874"},
        "required_phrases": ["2,500"],
    },
    {
        "name": "onshore student visa reapplication",
        "query": (
            "Can a person who already holds a Student visa subclass 500 "
            "and is in Australia apply for another Student visa as a primary applicant?"
        ),
        "required_versions": {"F2026L01347"},
        "required_phrases": ["1222(3)(aa)", "must be outside Australia"],
    },
    {
        "name": "defence student exception",
        "query": (
            "Can a Defence student who already holds a Student visa subclass 500 "
            "and is in Australia apply for another Student visa as a primary applicant?"
        ),
        "required_versions": {"F2026L01347"},
        "required_phrases": ["Defence students", "(3A)"],
    },
]


def test_retrieval_evaluation_cases():
    store = VectorStore()
    embedder = Embedder()

    failures = []

    for case in CASES:
        result = store.search(
            query_embedding=embedder.embed_query(case["query"]),
            top_k=5,
            query_text=case["query"],
        )

        versions = {
            str(metadata.get("version", ""))
            for metadata in result["metadatas"][0]
        }

        combined_text = " ".join(result["documents"][0]).lower()
        normalized_text = re.sub(r"[^a-z0-9]+", "", combined_text)

        missing_versions = case["required_versions"] - versions
        missing_phrases = []

        for phrase in case["required_phrases"]:
            normalized_phrase = re.sub(r"[^a-z0-9]+", "", phrase.lower())

            if normalized_phrase not in normalized_text:
                missing_phrases.append(phrase)

        if missing_versions or missing_phrases:
            failures.append(
                {
                    "case": case["name"],
                    "missing_versions": sorted(missing_versions),
                    "missing_phrases": missing_phrases,
                    "retrieved_versions": sorted(versions),
                }
            )

    assert not failures, (
        "Retrieval evaluation failures:\n"
        + "\n".join(str(item) for item in failures)
    )
