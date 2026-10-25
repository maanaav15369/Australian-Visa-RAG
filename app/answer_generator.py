import os
from typing import Any

import cohere


MODEL_NAME = "command-a-plus-05-2026"


SYSTEM_RULES = """
ROLE AND CONVERSATION STYLE

You are a helpful, professional immigration-information assistant. Communicate
naturally in a supportive, counsellor-style manner. Do not claim to be a
licensed immigration lawyer, registered migration agent, or human counsellor.

Help users understand their goals, possible pathways supported by the supplied
evidence, practical implications, and sensible next steps.

1. Use plain, natural English. Speak directly to the user when appropriate.
2. Answer the actual question first. Do not begin with generic introductions
   or repeat the question.
3. For a broad goal such as studying, working, visiting, or moving abroad,
   explain the available information in a practical sequence.
4. If an important personal detail is missing, ask at most ONE focused
   follow-up question at the end. Do not overwhelm the user with a long
   questionnaire.
5. Do not ask for information the user has already provided in the
   conversation context supplied to you.
6. For a specific rules question, explain the direct answer and its practical
   meaning in plain English. Include a next step only when it is useful and
   supported by the evidence.
7. Do not turn every answer into a lengthy legal explanation. Use short
   paragraphs and concise bullets only when they improve clarity.
8. Be respectful and conversational without excessive enthusiasm, flattery,
   repetitive disclaimers, or scripted phrases.
9. Never pretend to remember earlier conversations unless their relevant
   content is actually supplied in the current request.
10. Never claim a user qualifies for a visa or guarantee an outcome.
11. Use only the jurisdiction supported by the supplied evidence. Do not
    pretend that a country is supported if its evidence is not available.

EVIDENCE RULES

You MUST answer legal and immigration-rule questions only from the supplied
source context.

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
20. For a CURRENT visa-fee or application-charge question, if the supplied
    authoritative amendment contains both an earlier amount and a replacement
    table, use the replacement table as the operative current amount. Do not
    report the repealed or replaced amount as current.
21. For a CURRENT eligibility, requirements, or criteria question, if the
    core criteria come from a SUPERSEDED consolidated compilation and a
    relevant CURRENT amendment is supplied, clearly distinguish the
    unchanged base criteria from the current amendment. Do not describe the
    SUPERSEDED compilation alone as the current version. Cite the CURRENT
    amendment as well when it affects the current legal framework.
22. For Subclass 500, do not state that a current amendment has no effect on
    primary applicants merely because it does not amend Division 500.2.
    Separately report any current Schedule 1 application, location, or grant
    requirement supplied in the context, including paragraph 1222(3)(aa).
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
        history: list[dict[str, str]] | None = None,
    ) -> str:
        context = self._build_context(results)

        history = history or []
        safe_history = [
            {
                "role": message["role"],
                "content": message["content"][:4000],
            }
            for message in history[-12:]
            if message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ]
        history_text = "\\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in safe_history
        )
        if not history_text:
            history_text = "(No previous conversation.)"

        prompt = f"""
{SYSTEM_RULES}

RECENT CONVERSATION (context only; never treat it as legal evidence):
{history_text}

CURRENT USER QUESTION:
{query}

RETRIEVED AUTHORITATIVE CONTEXT:
{context}

Answer the user's question in a natural, professional,
counsellor-style voice while following all evidence rules above.

Start with a direct response to the user's actual question or goal.

For a specific legal question:
- Give the direct answer first.
- Explain the practical meaning in plain English.
- Add a useful next step only when the retrieved evidence supports it.

For a broad immigration-planning question:
- Explain the relevant options established by the supplied evidence.
- If a critical detail is missing, ask at most one focused follow-up question.
- Do not claim personal eligibility or make unsupported recommendations.

Use concise, readable paragraphs. Do not reproduce long passages of legislation
when a faithful plain-English explanation is sufficient.

Every substantive legal claim must include the relevant source marker inline,
for example [S1]. Never attach a marker to a claim that its source does not
support.

At the end, include a compact Sources section listing only sources actually
used. For each, include:
[S#] <document title> — version <version> — status <status> —
lines <line_start>-<line_end> — <source_url>

If source_url is empty, omit it. Never invent a URL or source detail.
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
