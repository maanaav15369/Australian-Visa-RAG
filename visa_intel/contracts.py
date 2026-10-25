"""Typed API contract for a jurisdiction-aware chat endpoint.

Not yet wired into api/main.py.
"""
import json
from datetime import date, timedelta
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DISCLAIMER = (
    "Information only, drawn from the indexed official sources. "
    "This is not legal advice and does not predict visa eligibility."
)

# A user may be a calendar day ahead of the server, so allow a small skew.
FUTURE_TOLERANCE_DAYS = 1


class Jurisdiction(str, Enum):
    AU = "AU"
    CA = "CA"
    NZ = "NZ"
    US = "US"


# Only add a code here once its collection exists AND its isolation tests pass.
AVAILABLE_JURISDICTIONS = frozenset({Jurisdiction.AU})


class JurisdictionUnavailable(Exception):
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__(
            "No indexed evidence yet for: " + ", ".join(j.value for j in self.missing)
        )


def ensure_available(jurisdictions):
    missing = [j for j in jurisdictions if j not in AVAILABLE_JURISDICTIONS]
    if missing:
        raise JurisdictionUnavailable(missing)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    jurisdictions: list[Jurisdiction] = Field(
        default_factory=lambda: [Jurisdiction.AU], min_length=1, max_length=4
    )
    as_of_date: date | None = None
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")

    @model_validator(mode="before")
    @classmethod
    def _accept_single_jurisdiction(cls, data: Any):
        if isinstance(data, dict) and "jurisdiction" in data:
            if "jurisdictions" in data:
                raise ValueError("Send either 'jurisdiction' or 'jurisdictions', not both.")
            data = dict(data)
            data["jurisdictions"] = [data.pop("jurisdiction")]
        return data

    @field_validator("question", mode="before")
    @classmethod
    def _strip_question(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("jurisdictions")
    @classmethod
    def _no_duplicates(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("Duplicate jurisdictions are not allowed.")
        return v

    @field_validator("as_of_date")
    @classmethod
    def _not_in_future(cls, v):
        if v is not None and v > date.today() + timedelta(days=FUTURE_TOLERANCE_DAYS):
            raise ValueError("as_of_date cannot be in the future.")
        return v

    @property
    def is_historical(self) -> bool:
        return self.as_of_date is not None


class Citation(BaseModel):
    """Every descriptive field is optional: show only what the evidence supports."""

    marker: str
    chunk_id: str
    jurisdiction: Jurisdiction
    title: str | None = None
    authority: str | None = None
    version: str | None = None
    legal_status: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    provision: str | None = None
    source_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "insufficient_evidence"]
    jurisdictions: list[Jurisdiction]
    as_of_date: date | None = None
    citations: list[Citation] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _answered_requires_citations(self):
        if self.status == "answered" and not self.citations:
            raise ValueError("An answered response must carry at least one citation.")
        return self


ErrorCode = Literal[
    "invalid_request",
    "jurisdiction_unavailable",
    "timeout",
    "rate_limited",
    "upstream_error",
    "internal_error",
]


class ErrorResponse(BaseModel):
    code: ErrorCode
    message: str


class StreamEventType(str, Enum):
    META = "meta"    # jurisdictions, as-of date, retrieved chunk ids; sent before text
    TOKEN = "token"  # provisional text delta
    DONE = "done"    # final text, validated citations, timings
    ERROR = "error"  # typed ErrorResponse payload


def format_sse(event, data):
    """Format one Server-Sent Event. `data` must already be JSON-serialisable."""
    name = event.value if isinstance(event, StreamEventType) else str(event)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n"
