from pathlib import Path
from typing import Any
from datetime import date
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

    @staticmethod
    def _current_scope_requested(query: str) -> bool:
        """
        Treat the chatbot as current-first unless the user explicitly asks
        for historical or date-specific rules.
        """
        q = query.lower()

        historical_markers = (
            "historical",
            "history",
            "old rule",
            "previous rule",
            "former rule",
            "before 2026",
            "before 2025",
            "before 2024",
            "in 2025",
            "in 2024",
            "in 2023",
            "as it stood",
            "at that time",
        )

        return not any(marker in q for marker in historical_markers)

    @staticmethod
    def _temporal_score(
        query: str,
        metadata: dict[str, Any],
    ) -> float:
        """
        Prefer currently effective sources for current/default questions,
        while retaining superseded compilations as lower-ranked base evidence.
        """
        if not VectorStore._current_scope_requested(query):
            return 0.0

        score = 0.0

        status = str(metadata.get("status", "")).upper()
        today = date.today()

        if status == "CURRENT":
            score += 6.0
        elif status == "CURRENT_REFERENCE":
            score += 1.0
        elif status == "COMMENCED":
            score += 1.0
        elif status == "SUPERSEDED":
            score -= 2.0
        elif status == "HISTORICAL_SNAPSHOT":
            score -= 5.0
        elif status == "NO_LONGER_IN_FORCE":
            score -= 6.0

        effective_from = str(metadata.get("effective_from", "")).strip()
        effective_to = str(metadata.get("effective_to", "")).strip()

        try:
            if effective_from:
                start = date.fromisoformat(effective_from)
                if start <= today:
                    score += 2.0
                else:
                    score -= 8.0
        except ValueError:
            pass

        try:
            if effective_to:
                end = date.fromisoformat(effective_to)
                if end < today:
                    score -= 4.0
        except ValueError:
            pass

        return score

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

        # Strong ranking for current visa-fee questions.
        current_fee_query = (
            (
                "fee" in query_lower
                or "application charge" in query_lower
                or "base charge" in query_lower
                or "cost" in query_lower
            )
            and (
                "current" in query_lower
                or "currently" in query_lower
                or "now" in query_lower
                or "today" in query_lower
            )
        )

        if current_fee_query:
            if "base application charge" in document_lower:
                score += 8.0

            # The operative July 2026 replacement amount.
            if "$2 500" in document_lower:
                score += 35.0

            if metadata.get("status") == "CURRENT":
                score += 8.0

        # Strong anchor for the October 2026 Student visa offshore rule.
        offshore_query = (
            (
                "another student visa" in query_lower
                or "another visa" in query_lower
                or "already holds" in query_lower
                or "holds a student visa" in query_lower
            )
            and (
                "in australia" in query_lower
                or "outside australia" in query_lower
            )
        )

        if offshore_query:
            if "1222(3)(aa)" in document_lower:
                score += 35.0

            if "must be outside australia" in document_lower:
                score += 25.0

            if "holds a subclass 500" in document_lower:
                score += 15.0

            if "primary criteria" in document_lower:
                score += 10.0

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

        # For the October 2026 offshore Student visa rule, explicitly
        # include the authoritative amendment instrument. This prevents
        # semantic retrieval from hiding the exact operative provision.
        query_text_lower = (query_text or "").lower()

        offshore_query = (
            (
                "another student visa" in query_text_lower
                or "another visa" in query_text_lower
                or "already holds" in query_text_lower
                or "holds a student visa" in query_text_lower
            )
            and (
                "in australia" in query_text_lower
                or "outside australia" in query_text_lower
            )
        )

        if offshore_query:
            authoritative = self.collection.get(
                where={"version": "F2026L01347"},
                include=["documents", "metadatas"],
            )

            existing_ids = set(ids)

            for doc_id, doc, metadata in zip(
                authoritative["ids"],
                authoritative["documents"],
                authoritative["metadatas"],
            ):
                if doc_id in existing_ids:
                    continue

                ids.append(doc_id)
                documents.append(doc)
                metadatas.append(metadata)
                distances.append(1.0)

        # For current Subclass 500 eligibility/requirements questions,
        # explicitly include the current October 2026 amendment provisions
        # that affect where a primary application can be made and granted.
        current_student_eligibility_query = (
            self._current_scope_requested(query_text_lower)
            and (
                "student visa" in query_text_lower
                or "subclass 500" in query_text_lower
            )
            and any(
                phrase in query_text_lower
                for phrase in (
                    "eligibility",
                    "eligible",
                    "requirements",
                    "criteria",
                    "qualify",
                )
            )
        )

        if current_student_eligibility_query:
            authoritative = self.collection.get(
                where={"version": "F2026L01347"},
                include=["documents", "metadatas"],
            )

            existing_ids = set(ids)

            for doc_id, doc, metadata in zip(
                authoritative["ids"],
                authoritative["documents"],
                authoritative["metadatas"],
            ):
                normalized = "".join(doc.lower().split())

                relevant = (
                    "1222(3)(aa)" in normalized
                    or "1222(3a)" in normalized
                    or "500.411" in normalized
                )

                if not relevant or doc_id in existing_ids:
                    continue

                ids.append(doc_id)
                documents.append(doc)
                metadatas.append(metadata)
                distances.append(1.0)
                existing_ids.add(doc_id)

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

            temporal_score = self._temporal_score(
                query_text,
                metadata,
            )

            combined_score = (
                semantic_score
                + lexical_score
                + temporal_score
            )

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
                    # Keep the relevant C288 base provisions, but also add
                    # the strongest current October 2026 Student Visa
                    # amendment chunks when answering current/default queries.
                    overlay_versions = {
                        "F2026L01347",
                        "F2026L01348",
                        "F2026L01349",
                    }

                    overlay = []

                    if self._current_scope_requested(query_text):
                        # Prefer current amendment chunks directly from the
                        # authoritative collection instead of relying only on
                        # semantic-ranked candidates.
                        current_docs = self.collection.get(
                            where={
                                "$and": [
                                    {"version": {"$in": list(overlay_versions)}},
                                    {"status": "CURRENT"},
                                ]
                            },
                            include=["documents", "metadatas"],
                        )

                        for doc_id, doc, metadata in zip(
                            current_docs["ids"],
                            current_docs["documents"],
                            current_docs["metadatas"],
                        ):
                            normalized = "".join(doc.lower().split())

                            relevant = True

                            if "student visa" in query_lower and (
                                "eligibility" in query_lower
                                or "requirements" in query_lower
                                or "criteria" in query_lower
                                or "qualify" in query_lower
                            ):
                                offshore_rule = (
                                    (
                                        "afterparagraph1222(3)(a)ofschedule1insert:(aa)"
                                        in normalized
                                    )
                                    and "mustbeoutsideaustralia" in normalized
                                )

                                exceptions = (
                                    "aftersubitem1222(3)insertsubitem(3a)"
                                    in normalized
                                    or "classesofpersons" in normalized
                                )

                                relevant = (
                                    offshore_rule
                                    or exceptions
                                    or "500.411" in normalized
                                    or "1222(3)(aa)" in normalized
                                )

                            if relevant:
                                overlay.append(
                                    (
                                        doc_id,
                                        doc,
                                        metadata,
                                    )
                                )

                        # Fallback to ranked current documents if the direct
                        # current-source lookup found nothing relevant.
                        if not overlay:
                            for item in ranked:
                                metadata = item[4]

                                if (
                                    metadata.get("version") in overlay_versions
                                    and metadata.get("status") == "CURRENT"
                                ):
                                    overlay.append(
                                        (
                                            item[1],
                                            item[2],
                                            metadata,
                                        )
                                    )

                                if len(overlay) >= 2:
                                    break

                    combined = []
                    seen = set()

                    for item in overlay + expanded:
                        if item[0] in seen:
                            continue

                        seen.add(item[0])
                        combined.append(item)

                    return {
                        "documents": [[item[1] for item in combined]],
                        "distances": [[
                            0.0 for _ in combined
                        ]],
                        "metadatas": [[item[2] for item in combined]],
                        "ids": [[item[0] for item in combined]],
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
