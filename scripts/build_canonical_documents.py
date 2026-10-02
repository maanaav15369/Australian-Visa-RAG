from pathlib import Path
import json

ROOT = Path(".")
REGISTRY = ROOT / "data/manifests/corpus_registry.json"
OUT = ROOT / "data/manifests/canonical_documents.json"

registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

docs = []

for r in registry:
    role = r.get("canonical_role")

    # Discovery material is never part of the authoritative corpus.
    if role == "discovery_only":
        continue

    # The duplicated Home Affairs HTML payload is retained only as reference.
    if role == "supplementary_reference":
        continue

    # Migration Regulations C288:
    # use the four extracted official compilation volumes.
    if r["document_id"] == "migration_regulations_1994_C288":
        volume_dir = ROOT / "data/raw/legislation/migration_regulations/C288"

        volumes = [
            "volume_1.txt",
            "volume_2.txt",
            "volume_3.txt",
            "volume_4_endnotes.txt",
        ]

        for volume in volumes:
            path = volume_dir / volume

            if not path.exists():
                raise FileNotFoundError(path)

            docs.append({
                "document_id": f"migration_regulations_1994_C288_{volume[:-4]}",
                "parent_source_id": r["document_id"],
                "title": f"{r['title']} — {volume[:-4].replace('_', ' ').title()}",
                "authority": r["authority"],
                "authority_level": r["authority_level"],
                "source_type": "legislation_compilation_volume",
                "status": r["status"],
                "version": r["version"],
                "effective_from": r["effective_from"],
                "effective_to": r.get("effective_to"),
                "retrieved_date": r.get("retrieved_date"),
                "source_url": r.get("url"),
                "local_file": str(path),
                "canonical_role": "authoritative"
            })

        continue

    # Normal single-file authoritative sources.
    path = r.get("local_file")

    if not path:
        continue

    p = Path(path)

    # Skip the raw Home Affairs directory or anything that is not a
    # directly ingestible canonical text source at this stage.
    if p.is_dir():
        continue

    docs.append(r)

# Add the amendment-block text files separately because they contain the
# actual 2026 amendment operations and are needed for legal provenance.
amendment_dir = ROOT / "data/processed/amendments"

for path in sorted(amendment_dir.glob("*.txt")):
    docs.append({
        "document_id": f"amendment_block_{path.stem}",
        "parent_source_id": path.stem,
        "title": path.stem.replace("_", " "),
        "authority": "Federal Register of Legislation",
        "authority_level": "tier_1",
        "source_type": "amendment_block",
        "status": "CURRENT_REFERENCE",
        "version": path.stem,
        "effective_from": None,
        "effective_to": None,
        "retrieved_date": "2026-10-02",
        "source_url": None,
        "local_file": str(path),
        "canonical_role": "amendment_evidence"
    })

OUT.write_text(
    json.dumps(docs, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Canonical documents:", len(docs))
print("Written:", OUT)
print()

from collections import Counter
counts = Counter(d["canonical_role"] for d in docs)

for role, count in sorted(counts.items()):
    print(f"{role}: {count}")

print()
print("=== DOCUMENTS ===")
for d in docs:
    print(
        f"{d['document_id']} | "
        f"{d['source_type']} | "
        f"{d['status']} | "
        f"{d['local_file']}"
    )
