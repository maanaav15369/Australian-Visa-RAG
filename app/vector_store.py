from pathlib import Path
from typing import Any
import re

import chromadb


CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "australian_visa_sources"


class VectorStore:
    """Store and search version-aware Australian visa source chunks."""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )

    @staticmethod
    def _metadata(record: dict[str, Any]) -> dict[str, Any]:
        fields = [
            "document_id",
            "chunk_index",
            "corpus",
            "title",
            "authority",
            "authority_level",
            "source_type",
            "canonical_role",
            "status",
            "version",
            "effective_from",
            "effective_to",
            "retrieved_date",
            "source_url",
            "local_file",
            "normalized_text_file",
            "line_start",
            "line_end",
            "character_count",
            "text_sha256",
        ]

        metadata: dict[str, Any] = {}

        for field in fields:
            value = record.get(field)

            if value is None:
                value = ""

            if isinstance(value, (str, int, float, bool)):
                metadata[field] = value
            else:
                metadata[field] = str(value)

        return metadata

    def add_records(
        self,
        records: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if len(records) != len(embeddings):
            raise ValueError(
                f"Records ({len(records)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        self.collection.upsert(
            ids=[record["chunk_id"] for record in records],
            documents=[record["text"] for record in records],
            embeddings=embeddings,
            metadatas=[self._metadata(record) for record in records],
        )

    @staticmethod
    def _legal_terms(query: str) -> list[str]:
        terms: list[str] = []

        subclasses = re.findall(
            r"\bsubclass\s+(\d{3})\b",
            query,
            flags=re.IGNORECASE,
        )

        for subclass in subclasses:
            terms.append(f"subclass {subclass}")
            terms.append(f"{subclass}.")

        clauses = re.findall(
            r"\b\d{1,3}\.\d{1,4}\b",
            query,
        )

        terms.extend(clauses)

        return [term.lower() for term in terms]

    @classmethod
    def _lexical_score(
        cls,
        query: str,
        document: str,
        metadata: dict[str, Any],
    ) -> float:
        query_lower = query.lower()
        document_lower = document.lower()
        score = 0.0

        # Exact legal identifiers.
        for term in cls._legal_terms(query):
            if term in document_lower:
                if term.startswith("subclass "):
                    score += 15.0
                elif term.endswith("."):
                    score += 8.0
                else:
                    score += 12.0

        # Eligibility-style questions should prefer criteria text.
        eligibility_query = any(
            phrase in query_lower
            for phrase in (
                "eligibility",
                "eligible",
                "requirements",
                "criteria",
                "qualify",
                "qualification",
            )
        )

        if eligibility_query:
            if "primary criteria" in document_lower:
                score += 8.0

            if "secondary criteria" in document_lower:
                score += 4.0

            for phrase in (
                "genuine applicant",
                "financial capacity",
                "health insurance",
                "english language",
            ):
                if phrase in document_lower:
                    score += 4.0

        # Exact topical phrases.
        for phrase in (
            "student visa",
            "primary criteria",
            "secondary criteria",
            "eligibility",
            "requirements",
        ):
            if phrase in query_lower and phrase in document_lower:
                score += 2.0

        # Prefer authoritative legislation for legal eligibility questions.
        if eligibility_query:
            source_type = str(
                metadata.get("source_type", "")
            ).lower()

            canonical_role = str(
                metadata.get("canonical_role", "")
            ).lower()

            if "legislation" in source_type:
                score += 4.0

            if canonical_role in {
                "authoritative",
                "authoritative_current_snapshot",
            }:
                score += 2.0

            # A Visa List is excellent for identifying available visas,
            # but is not the primary eligibility-criteria source.
            if source_type == "official_pdf_snapshot":
                score -= 5.0

        return score

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        query_text: str | None = None,
    ) -> Any:
        """Hybrid semantic + legal retrieval with criteria-section anchoring."""

        semantic_k = max(top_k * 20, 100)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": semantic_k,
        }

        if where is not None:
            kwargs["where"] = where

        semantic = self.collection.query(**kwargs)

        documents = semantic["documents"][0]
        distances = semantic["distances"][0]
        metadatas = semantic["metadatas"][0]
        ids = semantic["ids"][0]

        if not query_text:
            return {
                "documents": [documents[:top_k]],
                "distances": [distances[:top_k]],
                "metadatas": [metadatas[:top_k]],
                "ids": [ids[:top_k]],
            }

        query_lower = query_text.lower()

        ranked = []

        for doc_id, doc, distance, metadata in zip(
            ids,
            documents,
            distances,
            metadatas,
        ):
            semantic_score = 1.0 - float(distance)

            lexical_score = self._lexical_score(
                query_text,
                doc,
                metadata,
            )

            document_lower = doc.lower()

            # Strong anchor for a visa's primary-criteria section.
            if (
                "eligibility" in query_lower
                or "requirements" in query_lower
                or "criteria" in query_lower
                or "qualify" in query_lower
            ):
                if (
                    "subclass 500—student" in document_lower
                    and "500.2—primary criteria" in document_lower
                ):
                    lexical_score += 30.0

            combined_score = semantic_score + lexical_score

            ranked.append(
                (
                    combined_score,
                    doc_id,
                    doc,
                    distance,
                    metadata,
                )
            )

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # Find the strongest legal-section anchor.
        anchor = None

        if (
            "eligibility" in query_lower
            or "requirements" in query_lower
            or "criteria" in query_lower
            or "qualify" in query_lower
        ):
            for item in ranked:
                document_lower = item[2].lower()

                if (
                    "subclass 500—student" in document_lower
                    and "500.2—primary criteria" in document_lower
                    and item[4].get("source_type")
                    == "legislation_compilation_volume"
                ):
                    anchor = item
                    break

        if anchor is not None:
            anchor_metadata = anchor[4]
            document_id = anchor_metadata.get("document_id")
            anchor_index = anchor_metadata.get("chunk_index")

            if document_id and isinstance(anchor_index, int):
                expanded = []

                for neighbor_index in range(
                    anchor_index,
                    anchor_index + 3,
                ):
                    neighbor = self.collection.get(
                        where={
                            "$and": [
                                {"document_id": document_id},
                                {"chunk_index": neighbor_index},
                            ]
                        },
                        include=["documents", "metadatas"],
                    )

                    if not neighbor["ids"]:
                        continue

                    expanded.append(
                        (
                            neighbor["ids"][0],
                            neighbor["documents"][0],
                            neighbor["metadatas"][0],
                        )
                    )

                if expanded:
                    return {
                        "documents": [[item[1] for item in expanded]],
                        "distances": [[
                            0.0 for _ in expanded
                        ]],
                        "metadatas": [[item[2] for item in expanded]],
                        "ids": [[item[0] for item in expanded]],
                    }

        selected = ranked[:top_k]

        return {
            "documents": [[item[2] for item in selected]],
            "distances": [[item[3] for item in selected]],
            "metadatas": [[item[4] for item in selected]],
            "ids": [[item[1] for item in selected]],
        }

    def count(self) -> int:
        return self.collection.count()
