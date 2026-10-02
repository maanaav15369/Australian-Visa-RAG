from pathlib import Path

from app.chunker import create_chunks
from app.embedder import Embedder
from app.vector_store import VectorStore


SOURCE_PATH = Path("data/raw/home_affairs_visa_list.txt")


def main() -> None:
    text = SOURCE_PATH.read_text(encoding="utf-8")

    chunks = create_chunks(
        text=text,
        source="Department of Home Affairs - Visa List",
        chunk_size=1000,
        overlap=150,
    )

    print(f"Chunks created: {len(chunks)}")

    embedder = Embedder()

    document_texts = [chunk.text for chunk in chunks]

    print("Creating embeddings...")

    embeddings = embedder.embed_documents(document_texts)

    print("Storing vectors in ChromaDB...")

    vector_store = VectorStore()

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("\nIndexing complete.")
    print(f"Chunks stored: {len(chunks)}")


if __name__ == "__main__":
    main()