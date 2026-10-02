import json
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")


def save_document(document: dict[str, str]) -> tuple[Path, Path]:
    """Save document text and metadata separately."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    text_path = RAW_DATA_DIR / "home_affairs_visa_list.txt"
    metadata_path = RAW_DATA_DIR / "home_affairs_visa_list.json"

    text_path.write_text(
        document["text"],
        encoding="utf-8",
    )

    metadata = {
        "source": document["source"],
        "title": document["title"],
        "url": document["url"],
        "retrieved_at": document["retrieved_at"],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return text_path, metadata_path