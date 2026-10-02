import re
from dataclasses import dataclass


SECTION_HEADINGS = [
    "Visitor visas",
    "Studying and training visas",
    "Family and partner visas",
    "Working and skilled visas",
    "Refugee and humanitarian visas",
    "Other visas",
    "Repealed visas",
]


@dataclass
class Chunk:
    """A searchable piece of a source document."""

    chunk_id: str
    text: str
    source: str
    section: str


def normalize_heading(line: str) -> str:
    """Remove PDF table-of-contents anchor information."""
    cleaned_line = line.strip()

    cleaned_line = re.sub(
        r"\s*\(#content-index-\d+\)\s*$",
        "",
        cleaned_line,
    )

    return cleaned_line.strip()


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a visa-list document by major Home Affairs sections."""

    lines = text.splitlines()

    sections: list[tuple[str, list[str]]] = []

    current_section = "General"
    current_lines: list[str] = []

    for line in lines:
        cleaned_line = normalize_heading(line)

        if cleaned_line in SECTION_HEADINGS:
            if current_lines:
                sections.append(
                    (current_section, current_lines)
                )

            current_section = cleaned_line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            (current_section, current_lines)
        )

    result: list[tuple[str, str]] = []

    for section_name, section_lines in sections:
        section_text = "\n".join(section_lines).strip()

        if section_text:
            result.append(
                (section_name, section_text)
            )

    return result


def create_chunks(
    text: str,
    source: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Create searchable chunks while preserving section context."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    sections = split_into_sections(text)

    chunks: list[Chunk] = []
    chunk_number = 1

    for section_name, section_text in sections:

        # Skip the generic PDF/page introduction.
        if section_name == "General":
            continue

        start = 0

        while start < len(section_text):
            end = start + chunk_size

            raw_chunk_text = section_text[start:end].strip()

            if raw_chunk_text:
                chunk_text = (
                    f"Section: {section_name}\n\n"
                    f"{raw_chunk_text}"
                )

                chunks.append(
                    Chunk(
                        chunk_id=f"chunk-{chunk_number:04d}",
                        text=chunk_text,
                        source=source,
                        section=section_name,
                    )
                )

                chunk_number += 1

            start = end - overlap

    return chunks