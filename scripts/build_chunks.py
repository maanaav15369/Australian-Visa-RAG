from pathlib import Path
import json
import hashlib
import re

ROOT = Path(".")
MANIFEST = ROOT / "data/manifests/normalized_documents.json"
OUT_DIR = ROOT / "data/processed/chunks"

MAIN_OUT = OUT_DIR / "main_chunks.jsonl"
EVIDENCE_OUT = OUT_DIR / "evidence_chunks.jsonl"
CHUNK_MANIFEST = ROOT / "data/manifests/chunk_manifest.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CHARS = 1800
MAX_CHARS = 2400
OVERLAP_CHARS = 300


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_document(doc):
    retrieval_role = doc.get("retrieval_role")

    if retrieval_role == "main":
        return "main"

    if retrieval_role == "evidence":
        return "evidence"

    raise ValueError(
        f"Missing or invalid retrieval_role for {doc['document_id']}: "
        f"{retrieval_role!r}"
    )


def clean_line(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def split_oversized_block(block: str):
    """
    Split unusually large blocks on line boundaries first,
    then hard-split only when necessary.
    """
    if len(block) <= MAX_CHARS:
        return [block]

    lines = [clean_line(x) for x in block.splitlines() if clean_line(x)]

    pieces = []
    current = ""

    for line in lines:
        candidate = line if not current else current + "\n" + line

        if len(candidate) <= MAX_CHARS:
            current = candidate
            continue

        if current:
            pieces.append(current)

        if len(line) <= MAX_CHARS:
            current = line
        else:
            start = 0
            while start < len(line):
                pieces.append(line[start:start + MAX_CHARS])
                start += MAX_CHARS
            current = ""

    if current:
        pieces.append(current)

    return pieces


def make_chunks(text: str):
    """
    Paragraph-aware chunking with modest overlap.
    The original normalized text remains unchanged on disk.
    """
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    expanded = []
    for paragraph in paragraphs:
        expanded.extend(split_oversized_block(paragraph))

    chunks = []
    current_parts = []
    current_len = 0

    for paragraph in expanded:
        add_len = len(paragraph) + (1 if current_parts else 0)

        if current_parts and current_len + add_len > TARGET_CHARS:
            chunk = "\n\n".join(current_parts)
            chunks.append(chunk)

            # Build overlap from the end of the previous chunk.
            overlap = chunk[-OVERLAP_CHARS:].strip()

            current_parts = [overlap, paragraph] if overlap else [paragraph]
            current_len = sum(len(x) for x in current_parts) + 2
        else:
            current_parts.append(paragraph)
            current_len += add_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def line_range(full_text: str, chunk_text: str, search_from: int):
    pos = full_text.find(chunk_text[:120], search_from)

    if pos == -1:
        pos = search_from

    start_line = full_text.count("\n", 0, pos) + 1
    end_pos = min(len(full_text), pos + len(chunk_text))
    end_line = full_text.count("\n", 0, end_pos) + 1

    return start_line, end_line, pos


docs = json.loads(MANIFEST.read_text(encoding="utf-8"))

main_records = []
evidence_records = []

document_summary = []

for doc in docs:
    text_path = ROOT / doc["normalized_text_file"]

    if not text_path.exists():
        print(f"SKIP MISSING TEXT: {doc['document_id']}")
        continue

    text = text_path.read_text(encoding="utf-8").strip()

    corpus = classify_document(doc)
    chunks = make_chunks(text)

    search_from = 0

    for index, chunk_text in enumerate(chunks):
        chunk_id = hashlib.sha256(
            f"{doc['document_id']}::{index}::{chunk_text}".encode("utf-8")
        ).hexdigest()[:24]

        start_line, end_line, found_pos = line_range(
            text,
            chunk_text,
            search_from,
        )

        search_from = max(found_pos + 1, search_from)

        record = {
            "chunk_id": chunk_id,
            "document_id": doc["document_id"],
            "chunk_index": index,
            "corpus": corpus,

            "title": doc.get("title"),
            "authority": doc.get("authority"),
            "authority_level": doc.get("authority_level"),

            "source_type": doc.get("source_type"),
            "canonical_role": doc.get("canonical_role"),
            "status": doc.get("status"),

            "version": doc.get("version"),
            "effective_from": doc.get("effective_from"),
            "effective_to": doc.get("effective_to"),
            "retrieved_date": doc.get("retrieved_date"),

            "source_url": doc.get("source_url"),
            "local_file": doc.get("local_file"),
            "normalized_text_file": doc.get("normalized_text_file"),

            "line_start": start_line,
            "line_end": end_line,

            "character_count": len(chunk_text),
            "text_sha256": sha256_text(chunk_text),

            "text": chunk_text,
        }

        if corpus == "main":
            main_records.append(record)
        else:
            evidence_records.append(record)

    document_summary.append({
        "document_id": doc["document_id"],
        "corpus": corpus,
        "document_status": doc.get("status"),
        "canonical_role": doc.get("canonical_role"),
        "chunks": len(chunks),
        "characters": len(text),
    })

with MAIN_OUT.open("w", encoding="utf-8") as f:
    for record in main_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with EVIDENCE_OUT.open("w", encoding="utf-8") as f:
    for record in evidence_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

manifest = {
    "chunking_version": "1.0",
    "target_chars": TARGET_CHARS,
    "max_chars": MAX_CHARS,
    "overlap_chars": OVERLAP_CHARS,
    "main_chunk_count": len(main_records),
    "evidence_chunk_count": len(evidence_records),
    "documents": document_summary,
    "main_output": str(MAIN_OUT),
    "evidence_output": str(EVIDENCE_OUT),
}

CHUNK_MANIFEST.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("=== CHUNKING COMPLETE ===")
print("Documents:", len(document_summary))
print("Main chunks:", len(main_records))
print("Evidence chunks:", len(evidence_records))
print("Main:", MAIN_OUT)
print("Evidence:", EVIDENCE_OUT)
print("Manifest:", CHUNK_MANIFEST)

print()
print("=== DOCUMENT CHUNK COUNTS ===")

for item in document_summary:
    print(
        f"{item['document_id']} | "
        f"{item['corpus']} | "
        f"{item['chunks']} chunks | "
        f"{item['characters']:,} chars"
    )
