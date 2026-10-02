from pathlib import Path
import json
from datetime import date

ROOT = Path(".")
OUT = ROOT / "data" / "manifests" / "corpus_registry.json"

manifest = json.loads(
    (ROOT / "data/manifests/source_manifest.json").read_text(encoding="utf-8")
)

registry = []

# Authoritative entries from the source manifest.
for src in manifest:
    authority_level = src.get("authority_level")

    if authority_level == "tier_1":
        registry.append({
            "document_id": src["source_id"],
            "title": src["title"],
            "authority": src["authority"],
            "authority_level": authority_level,
            "source_type": src["source_type"],
            "status": src["status"],
            "version": src.get("version"),
            "effective_from": src.get("effective_from"),
            "effective_to": src.get("effective_to"),
            "retrieved_date": src.get("retrieved_date"),
            "source_url": src.get("url"),
            "local_file": src.get("local_file"),
            "canonical_role": "authoritative"
        })

# Derived text from the 2026 Federal Register support PDFs.
support_dir = ROOT / "data/processed/support_pdfs"

support_roles = {
    "F2024L01618_subclass_186_current_compilation.txt":
        "current_compilation",
    "F2024L01620_subclass_482_current_compilation.txt":
        "current_compilation",
    "F2019L00278_general_skilled_occupations_current_compilation.txt":
        "current_compilation",
    "F2019L01405_subclass_494_current_compilation.txt":
        "current_compilation",
    "F2025L00904_LIN25-014_legislative_instrument.txt":
        "legislative_instrument_support",
    "F2025L00904_LIN25-014_explanatory_statement.txt":
        "explanatory_statement",
    "F2025L00905_LIN25-016_legislative_instrument.txt":
        "legislative_instrument_support",
    "F2025L00905_LIN25-016_explanatory_statement.txt":
        "explanatory_statement",
    "F2026L00369_legislative_instrument.txt":
        "legislative_instrument_support",
    "F2026L00369_explanatory_statement.txt":
        "explanatory_statement",
}

for filename, role in support_roles.items():
    path = support_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing support text: {path}")

    registry.append({
        "document_id": path.stem,
        "title": path.stem.replace("_", " "),
        "authority": "Federal Register of Legislation",
        "authority_level": "tier_1",
        "source_type": "support_document",
        "status": "CURRENT_REFERENCE",
        "version": path.stem,
        "effective_from": None,
        "effective_to": None,
        "retrieved_date": str(date(2026, 10, 2)),
        "source_url": None,
        "local_file": str(path),
        "canonical_role": role
    })

# Explicitly identify the Home Affairs material without treating every
# duplicated HTML payload as an independent source.
registry.append({
    "document_id": "home_affairs_current_pages_2026-10-02",
    "title": "Department of Home Affairs current visa/support pages",
    "authority": "Department of Home Affairs",
    "authority_level": "tier_1",
    "source_type": "web_pages",
    "status": "CURRENT_OPERATIONAL_REFERENCE",
    "version": "retrieved-2026-10-02",
    "effective_from": None,
    "effective_to": None,
    "retrieved_date": "2026-10-02",
    "source_url": "https://immi.homeaffairs.gov.au/",
    "local_file": "data/raw/home_affairs/current/",
    "canonical_role": "supplementary_reference",
    "index_individually": False,
    "notes": "Raw pages differ, but extracted common embedded payload is duplicated across files."
})

# Discovery material is deliberately registered but excluded from authoritative retrieval.
registry.append({
    "document_id": "visa_knowledge_base_May-2026",
    "title": "Australia Complete Visa Knowledge Base",
    "authority": "Department of Home Affairs attribution in supplied document",
    "authority_level": "discovery_only",
    "source_type": "docx",
    "status": "DISCOVERY_ONLY",
    "version": "May-2026",
    "effective_from": None,
    "effective_to": None,
    "retrieved_date": None,
    "source_url": None,
    "local_file": "data/raw/discovery/Australia_Complete_Visa_Knowledge_Base.docx",
    "canonical_role": "discovery_only",
    "index_individually": False
})

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(registry, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Registry written:", OUT)
print("Total registry entries:", len(registry))

from collections import Counter
counts = Counter(r["canonical_role"] for r in registry)

print()
print("By canonical role:")
for role, count in sorted(counts.items()):
    print(f"  {role}: {count}")

print()
print("Retrieval candidates:")
for r in registry:
    if r.get("canonical_role") in {
        "authoritative",
        "current_compilation",
        "legislative_instrument_support",
        "explanatory_statement",
    }:
        print(" ", r["document_id"])
