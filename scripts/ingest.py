from app.source_loader import load_home_affairs_visa_list
from app.source_storage import save_document


def main() -> None:
    print("Loading Home Affairs visa list...")

    document = load_home_affairs_visa_list()

    print("Saving document...")

    text_path, metadata_path = save_document(document)

    print("\nIngestion complete.")
    print(f"Text saved to: {text_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()