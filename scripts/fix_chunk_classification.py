from pathlib import Path
import json

ROOT = Path(".")

MANIFEST = ROOT / "data/manifests/normalized_documents.json"

AMENDMENT_IDS = {
    "migration_legislation_migration_agents_F2026L00120",
    "migration_amendment_temporary_graduate_charge_F2026L00163",
    "migration_amendment_training_visas_F2026L00221",
    "air_navigation_legislation_F2026L00224",
    "migration_amendment_admin_review_student_F2026L00640",
    "migration_amendment_community_support_F2026L00733",
    "home_affairs_legislation_measures_2026_F2026L00874",
    "migration_amendment_working_holiday_age_F2026L00876",
}

docs = json.loads(MANIFEST.read_text(encoding="utf-8"))

changed = 0

for d in docs:
    doc_id = d["document_id"]

    if doc_id in AMENDMENT_IDS:
        d["canonical_role"] = "amendment_evidence"
        d["retrieval_role"] = "evidence"
        changed += 1

    elif doc_id == "migration_agents_regulations_2026_F2026L00118":
        d["retrieval_role"] = "main"

    elif d.get("document_id") == "visa_list_homeaffairs_2026-10-01":
        d["retrieval_role"] = "evidence"

    elif d.get("status") == "CURRENT_METADATA_ONLY":
        d["retrieval_role"] = "evidence"

    else:
        role = d.get("canonical_role", "")
        if role in {
            "authoritative",
            "authoritative_current_snapshot",
            "current_compilation",
        }:
            d["retrieval_role"] = "main"
        else:
            d["retrieval_role"] = "evidence"

MANIFEST.write_text(
    json.dumps(docs, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Updated documents:", changed)

print()
print("=== RETRIEVAL ROLE CHECK ===")

for d in docs:
    print(
        f"{d['document_id']} | "
        f"{d.get('status')} | "
        f"{d.get('canonical_role')} | "
        f"{d.get('retrieval_role')}"
    )
