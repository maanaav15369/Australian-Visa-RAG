from __future__ import annotations

import os
from typing import Any

from google import genai


MODEL_NAME = "gemini-3.5-flash"


SYSTEM_RULES = """
You are an Australian visa legal-information RAG answer generator.

You MUST answer only from the supplied source context.

Rules:
1. Do not use outside knowledge.
2. Do not invent, infer, or fill missing legal requirements.
3. Treat CURRENT authoritative legislation and official current snapshots
   as the primary evidence.
4. Treat amendment instruments and other evidence documents as supporting
   evidence, not as a replacement for the current consolidated law.
5. Never silently combine different versions.
6. When sources conflict, explicitly state the conflict and identify the
   versions involved.
7. Every substantive legal statement must include one or more source
   markers such as [S1] or [S2].
8. When the supplied sources do not establish an answer, say exactly:
   "Insufficient authoritative evidence found."
9. Distinguish legal requirements from explanatory commentary.
10. Answer only the scope of the user's question. Do not add secondary
    criteria, family-member criteria, visa conditions, fees, or other
    provisions unless the user asks about them.
11. Do not infer that a neighboring provision is part of the answer merely
    because it appears in the retrieved context.
12. Do not provide personalized legal advice or certainty about an
    individual's eligibility.
13. Keep the answer structured and concise.
14. Preserve visa subclass numbers and legal provision numbers exactly.
"""


class AnswerGenerator:
    """Generate grounded answers from retrieved RAG context."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(api_key=api_key)

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

For each substantive statement, append the relevant source marker,
for example [S1].

At the end, include:

Sources:
For every source actually used, include:
[S#] <document title> — version <version> — status <status> —
lines <line_start>-<line_end> — <source_url>

If source_url is empty, omit it.

Only list sources actually used.
"""

        interaction = self.client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
            generation_config={
                "thinking_level": "low",
            },
        )

        output = interaction.output_text

        if not output:
            return "Insufficient authoritative evidence found."

        return output
