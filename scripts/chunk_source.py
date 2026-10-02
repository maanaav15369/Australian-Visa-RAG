from pathlib import Path

from app.chunker import create_chunks


SOURCE_PATH = Path("data/raw/home_affairs_visa_list.txt")


def main() -> None:
    text = SOURCE_PATH.read_text(encoding="utf-8")

    chunks = create_chunks(
        text=text,
        source="Department of Home Affairs - Visa List",
        chunk_size=1000,
        overlap=150,
    )

    print(f"Document characters: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n" + "=" * 80)
        print(f"ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print("-" * 80)
        print(chunk.text)


if __name__ == "__main__":
    main()