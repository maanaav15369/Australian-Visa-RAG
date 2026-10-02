from __future__ import annotations

import json
from pathlib import Path

from app.embedder import Embedder
from app.vector_store import VectorStore


CHUNK_FILES = [
    Path("data/processed/chunks/main_chunks.jsonl"),
    Path("data/processed/chunks/evidence_chunks.jsonl"),
]

EMBED_BATCH_SIZE = 64
CHROMA_BATCH_SIZE = 100


def load_records() -> list[dict]:
    records: list[dict] = []

    for path in CHUNK_FILES:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    return records


def main() -> None:
    print("Loading validated chunk corpus...")

    records = load_records()

    expected = 1919
    if len(records) != expected:
        raise RuntimeError(
            f"Expected {expected} chunks, but found {len(records)}."
        )

    chunk_ids = [record["chunk_id"] for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("Duplicate chunk_id values detected.")

    main_count = sum(record.get("corpus") == "main" for record in records)
    evidence_count = sum(
        record.get("corpus") == "evidence" for record in records
    )

    print(f"Main chunks: {main_count}")
    print(f"Evidence chunks: {evidence_count}")
    print(f"Total chunks: {len(records)}")

    store = VectorStore()

    existing = store.count()
    if existing != 0:
        raise RuntimeError(
            f"Chroma collection already contains {existing} vectors. "
            "Refusing to overwrite."
        )

    print("\nLoading embedding model...")
    embedder = Embedder()

    print("\nCreating embeddings...")
    for start in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[start:start + EMBED_BATCH_SIZE]
        texts = [record["text"] for record in batch]

        embeddings = embedder.embed_documents(texts)

        print(
            f"Storing chunks {start + 1}-{start + len(batch)} "
            f"of {len(records)}..."
        )

        for store_start in range(0, len(batch), CHROMA_BATCH_SIZE):
            store_batch = batch[
                store_start:store_start + CHROMA_BATCH_SIZE
            ]
            store_embeddings = embeddings[
                store_start:store_start + CHROMA_BATCH_SIZE
            ]

            store.add_records(
                records=store_batch,
                embeddings=store_embeddings,
            )

    final_count = store.count()

    print("\nIndexing complete.")
    print(f"Expected vectors: {len(records)}")
    print(f"Stored vectors:   {final_count}")

    if final_count != len(records):
        raise RuntimeError(
            f"Vector count mismatch: expected {len(records)}, "
            f"stored {final_count}."
        )


if __name__ == "__main__":
    main()
