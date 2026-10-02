from app.answer_generator import AnswerGenerator
from app.embedder import Embedder
from app.vector_store import VectorStore


def answer_question(query: str) -> str:
    embedder = Embedder()
    store = VectorStore()
    generator = AnswerGenerator()

    results = store.search(
        query_embedding=embedder.embed_query(query),
        query_text=query,
        top_k=3,
        where={"corpus": "main"},
    )

    return generator.answer(
        query=query,
        results=results,
    )


def main() -> None:
    print("Australian Visa RAG Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        try:
            answer = answer_question(query)
            print("\nAssistant:")
            print(answer)
            print()
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
