from app.embedder import Embedder
from app.vector_store import VectorStore


def main() -> None:
    query = "What are the eligibility requirements for the Student visa subclass 500?"

    print(f"Query: {query}")

    embedder = Embedder()
    query_embedding = embedder.embed_query(query)

    vector_store = VectorStore()

    results = vector_store.search(
        query_embedding=query_embedding,
        query_text=query,
        top_k=3,
        where={"corpus": "main"},
    )

    print("\nTop results:\n")

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    for index, (
        chunk_id,
        document,
        distance,
        metadata,
    ) in enumerate(
        zip(ids, documents, distances, metadatas),
        start=1,
    ):
        print("=" * 80)
        print(f"RESULT {index}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Distance: {distance}")
        print(f"Document: {metadata.get('document_id')}")
        print(f"Version: {metadata.get('version')}")
        print(f"Status: {metadata.get('status')}")
        print(f"Effective from: {metadata.get('effective_from')}")
        print("-" * 80)
        print(document[:1200])
        print()


if __name__ == "__main__":
    main()
