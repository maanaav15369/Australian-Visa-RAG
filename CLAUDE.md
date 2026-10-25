# Project rules

Multi-country immigration RAG chatbot (AU first; CA, NZ, US planned).
Stack: Python 3.14, FastAPI, Streamlit, ChromaDB, sentence-transformers, Cohere.

## Hard rules
- Never read, print or commit `.env` or any API key.
- Never push, force-push, reset or delete branches. Commit locally only.
- Never hard-code answers, citations, dates or legal text.
- Do not change retrieval ranking without a measured reason and a regression test.
- Run `python -m pytest -q` after every change and show the real output.
- Baseline: 3 tests passed before changes. Do not break them.
- Single-country queries must never retrieve another country's evidence.
- The exact fallback text is: Insufficient authoritative evidence found.

## Measured facts
- Warm retrieval is about 20 ms. Do not optimise or cache retrieval.
- Embedder init was 6.4 s online vs 0.18 s with HF_HUB_OFFLINE=1.
- Cohere SDK 7.2.0; answer_generator.py uses a non-streaming ClientV2 chat call.

## Working style
Small commits. Say what changed, why, and what you ran.
