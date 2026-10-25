"""Turn the model's [S#] markers into structured citations.

Pure functions: no model, no network.
"""
import re
from typing import Any

from visa_intel.contracts import Citation, Jurisdiction

FALLBACK_TEXT = "Insufficient authoritative evidence found."

_BRACKET = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_NUMBER = re.compile(r"S(\d+)")


def is_insufficient(answer: str) -> bool:
    return answer.strip().startswith(FALLBACK_TEXT)


def _clean(value: Any):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _line_range(meta: dict):
    start, end = _clean(meta.get("line_start")), _clean(meta.get("line_end"))
    return f"lines {start}-{end}" if start and end else None


def extract_citations(answer: str, results: dict, jurisdiction: Jurisdiction):
    """One Citation per distinct marker that points at a really retrieved chunk.

    Markers pointing outside the retrieved set (e.g. [S9] with 5 results)
    are ignored, never invented.
    """
    ids = (results.get("ids") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    seen: set[int] = set()
    citations = []
    for group in _BRACKET.findall(answer):
        for raw in _NUMBER.findall(group):
            number = int(raw)
            if number in seen:
                continue
            seen.add(number)
            idx = number - 1
            if not 0 <= idx < len(ids):
                continue
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            citations.append(
                Citation(
                    marker=f"S{number}",
                    chunk_id=str(ids[idx]),
                    jurisdiction=jurisdiction,
                    title=_clean(meta.get("title")),
                    authority=_clean(meta.get("authority")),
                    version=_clean(meta.get("version")),
                    legal_status=_clean(meta.get("status")),
                    effective_from=_clean(meta.get("effective_from")),
                    effective_to=_clean(meta.get("effective_to")),
                    provision=_line_range(meta),
                    source_url=_clean(meta.get("source_url")),
                )
            )
    return citations
