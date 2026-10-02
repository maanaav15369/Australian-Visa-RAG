from pathlib import Path
import json
import re
import shutil
import subprocess
from bs4 import BeautifulSoup

ROOT = Path(".")
REGISTRY = ROOT / "data/manifests/canonical_documents.json"
OUT = ROOT / "data/processed/canonical_text"
OUT.mkdir(parents=True, exist_ok=True)

docs = json.loads(REGISTRY.read_text(encoding="utf-8"))

def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines but preserve document structure.
    lines = []
    previous_blank = False

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
        else:
            lines.append(line)
            previous_blank = False

    return "\n".join(lines).strip() + "\n"

def html_to_text(path: Path) -> str:
    raw = path.read_text(errors="replace", encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return normalize_text(soup.get_text("\n"))

def pdf_to_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return normalize_text(result.stdout)

def txt_to_text(path: Path) -> str:
    return normalize_text(path.read_text(errors="replace", encoding="utf-8"))

def convert_source(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return pdf_to_text(path)

    if suffix in {".html", ".htm"}:
        return html_to_text(path)

    if suffix == ".txt":
        return txt_to_text(path)

    if suffix == ".json":
        return normalize_text(path.read_text(errors="replace", encoding="utf-8"))

    raise ValueError(f"Unsupported source type: {path}")

written = 0
failed = 0

manifest_out = []

for doc in docs:
    source = ROOT / doc["local_file"]

    # Every canonical document is expected to point to one file at this
    # stage. Directory-backed supplementary material was excluded earlier.
    if source.is_dir():
        print(f"SKIP DIRECTORY: {doc['document_id']}")
        continue

    if not source.exists():
        print(f"FAILED MISSING: {doc['document_id']} -> {source}")
        failed += 1
        continue

    try:
        text = convert_source(source)

        if len(text.strip()) < 100:
            print(f"FAILED TOO SHORT: {doc['document_id']} ({len(text)} chars)")
            failed += 1
            continue

        out = OUT / f"{doc['document_id']}.txt"
        out.write_text(text, encoding="utf-8")

        item = dict(doc)
        item["normalized_text_file"] = str(out)
        item["character_count"] = len(text)
        item["line_count"] = len(text.splitlines())

        manifest_out.append(item)

        print(
            f"OK  {doc['document_id']} | "
            f"{len(text):,} chars | "
            f"{len(text.splitlines()):,} lines"
        )

        written += 1

    except Exception as exc:
        print(f"FAILED {doc['document_id']}: {exc}")
        failed += 1

out_manifest = ROOT / "data/manifests/normalized_documents.json"
out_manifest.write_text(
    json.dumps(manifest_out, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print()
print("=== NORMALIZATION SUMMARY ===")
print("Written:", written)
print("Failed:", failed)
print("Manifest:", out_manifest)
print("Output:", OUT)
