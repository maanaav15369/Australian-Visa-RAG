from pathlib import Path

from app.chunker import create_chunks
from app.embedder import Embedder


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

    texts = [chunk.text for chunk in chunks[:5]]

    embeddings = embedder.embed_documents(texts)

    print(f"Embeddings created: {len(embeddings)}")
    print(f"Vector dimension: {len(embeddings[0])}")

    print("\nFirst 10 values:")
    print(embeddings[0][:10])


if __name__ == "__main__":
    main()