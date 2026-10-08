import os
from typing import Any

import cohere


MODEL_NAME = "command-a-plus-05-2026"


SYSTEM_RULES = """
You are an Australian visa legal-information RAG answer generator.

You MUST answer only from the supplied source context.

Rules:
1. Do not use outside knowledge.
2. Do not invent, infer, or fill missing legal requirements.
3. Treat CURRENT authoritative legislation and official current sources
   as primary evidence.
4. For rules effective after the latest available consolidated compilation,
   reconstruct the effective rule only from authoritative evidence supplied
   in the context. A superseded compilation may provide unchanged base text,
   while a current amendment or legislative instrument supplies the operative
   change.
5. Never present a superseded compilation itself as the current version.
6. When reconstructing a current rule, distinguish the base compilation from
   the current amendment/instrument and respect effective_from and
   effective_to dates.
7. For questions containing "current", "currently", "now", or "today", do
   not answer solely from a SUPERSEDED, HISTORICAL_SNAPSHOT, or
   NO_LONGER_IN_FORCE source when a relevant CURRENT source is supplied.
8. If a CURRENT amendment or legislative instrument directly changes the
   provision being discussed, explicitly describe that operative change and
   cite the CURRENT source.
9. A SUPERSEDED compilation may provide unchanged base criteria, but a
   current answer must also cite the applicable CURRENT source when one is
   supplied.
10. Never silently combine different versions.
11. When sources conflict, explicitly identify the conflict and versions.
12. Every substantive legal statement must include one or more source markers
    such as [S1] or [S2].
13. When the supplied sources do not establish an answer, say exactly:
    "Insufficient authoritative evidence found."
14. Distinguish legal requirements from explanatory commentary.
15. Answer only the scope of the user's question.
16. If the user asks for "requirements", "eligibility", or "criteria" for
    a visa subclass without specifying secondary applicants, answer the
    PRIMARY criteria only. Do not add secondary criteria, family-member
    criteria, visa conditions, fees, or other provisions unless explicitly
    requested.
17. Do not provide personalized legal advice or certainty about an
    individual's eligibility.
18. Keep the answer structured and concise.
19. Preserve visa subclass numbers and legal provision numbers exactly.
"""



class AnswerGenerator:
    """Generate grounded answers from retrieved RAG context using Cohere."""

    def __init__(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not configured."
            )

        self.client = cohere.ClientV2(api_key=api_key)

    @staticmethod
    def _build_context(
        results: dict[str, Any],
    ) -> str:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        ids = results["ids"][0]

        blocks = []

        for index, (chunk_id, document, metadata) in enumerate(
            zip(ids, documents, metadatas),
            start=1,
        ):
            source_label = f"S{index}"

            blocks.append(
                f"""[{source_label}]
chunk_id: {chunk_id}
document_id: {metadata.get("document_id", "")}
title: {metadata.get("title", "")}
authority: {metadata.get("authority", "")}
authority_level: {metadata.get("authority_level", "")}
source_type: {metadata.get("source_type", "")}
canonical_role: {metadata.get("canonical_role", "")}
status: {metadata.get("status", "")}
version: {metadata.get("version", "")}
effective_from: {metadata.get("effective_from", "")}
effective_to: {metadata.get("effective_to", "")}
retrieved_date: {metadata.get("retrieved_date", "")}
line_start: {metadata.get("line_start", "")}
line_end: {metadata.get("line_end", "")}
source_url: {metadata.get("source_url", "")}

TEXT:
{document}
"""
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response.message, "content", None) or []

        text_parts = []

        for item in content:
            if getattr(item, "type", None) == "text":
                value = getattr(item, "text", None)
                if value:
                    text_parts.append(value)

        return "\n".join(text_parts).strip()

    def answer(
        self,
        query: str,
        results: dict[str, Any],
    ) -> str:
        context = self._build_context(results)

        prompt = f"""
{SYSTEM_RULES}

USER QUESTION:
{query}

RETRIEVED AUTHORITATIVE CONTEXT:
{context}

Answer the user's question using only the retrieved context.

Start directly with the answer.

For every substantive statement, append the relevant source marker,
for example [S1].

At the end, include:

Sources:
For every source actually used, include:
[S#] <document title> — version <version> — status <status> —
lines <line_start>-<line_end> — <source_url>

If source_url is empty, omit it.

Only list sources actually used.
"""

        response = self.client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful Australian visa legal-information "
                        "RAG answer generator. Follow the user's supplied "
                        "retrieved evidence exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        output = self._extract_text(response)

        if not output:
            return "Insufficient authoritative evidence found."

        return output
