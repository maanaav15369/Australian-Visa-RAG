from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


PDF_SOURCE_PATH = Path("data/raw/visa_list_home_affairs.pdf")


def load_home_affairs_visa_list() -> dict[str, str]:
    """Extract text from the official Home Affairs visa-list PDF."""

    if not PDF_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"PDF source not found: {PDF_SOURCE_PATH}"
        )

    reader = PdfReader(PDF_SOURCE_PATH)

    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            pages.append(
                f"\n--- Page {page_number} ---\n"
                f"{page_text}"
            )

    if not pages:
        raise RuntimeError(
            "No text could be extracted from the PDF."
        )

    text = "\n".join(pages)

    return {
        "source": "Department of Home Affairs",
        "title": "Visa list",
        "url": (
            "https://immi.homeaffairs.gov.au/"
            "visas/getting-a-visa/visa-listing"
        ),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
    }