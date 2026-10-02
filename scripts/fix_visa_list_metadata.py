from pathlib import Path
import json

ROOT = Path(".")

FILES = [
    ROOT / "data/manifests/canonical_documents.json",
    ROOT / "data/manifests/normalized_documents.json",
]

def patch_docs(path):
    docs = json.loads(path.read_text(encoding="utf-8"))

    for d in docs:
        doc_id = d.get("document_id", "")

        if doc_id == "visa_list_homeaffairs_2026-09-30":
            d["status"] = "CURRENT_SNAPSHOT"
            d["effective_from"] = "2026-09-30"
            d["effective_to"] = None
            d["canonical_role"] = "authoritative_current_snapshot"
            d["index_for_substantive_retrieval"] = True
            d["notes"] = (
                "Official Department of Home Affairs PDF snapshot captured "
                "30 September 2026. Contains the substantive visa catalogue."
            )

        elif doc_id == "visa_list_homeaffairs_2026-10-01":
            d["status"] = "CURRENT_METADATA_ONLY"
            d["canonical_role"] = "supplementary_metadata"
            d["index_for_substantive_retrieval"] = False
            d["notes"] = (
                "Official Department of Home Affairs webpage. The downloaded "
                "HTML is a shell and does not contain the substantive visa "
                "catalogue in parsed text. Retained for provenance and page "
                "metadata such as last-updated information."
            )

    path.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Updated: {path}")

for path in FILES:
    if path.exists():
        patch_docs(path)

# Patch corpus registry separately because its schema differs.
registry_path = ROOT / "data/manifests/corpus_registry.json"

if registry_path.exists():
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    for d in registry:
        doc_id = d.get("source_id") or d.get("document_id") or ""

        if doc_id == "visa_list_homeaffairs_2026-09-30":
            d["status"] = "CURRENT_SNAPSHOT"
            d["effective_from"] = "2026-09-30"
            d["effective_to"] = None
            d["index_for_substantive_retrieval"] = True
            d["notes"] = (
                "Official Home Affairs PDF snapshot; substantive visa catalogue."
            )

        elif doc_id == "visa_list_homeaffairs_2026-10-01":
            d["status"] = "CURRENT_METADATA_ONLY"
            d["index_for_substantive_retrieval"] = False
            d["notes"] = (
                "Official Home Affairs webpage shell; retained for metadata "
                "and provenance, not substantive retrieval."
            )

    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Updated: {registry_path}")

print()
print("=== VISA LIST METADATA FIX COMPLETE ===")
