import streamlit as st

from app.answer_generator import AnswerGenerator
from app.embedder import Embedder
from app.vector_store import VectorStore


st.set_page_config(
    page_title="Australian Visa RAG",
    page_icon="🇦🇺",
    layout="wide",
)


@st.cache_resource
def load_components():
    embedder = Embedder()
    store = VectorStore()
    generator = AnswerGenerator()
    return embedder, store, generator


def answer_question(query: str) -> str:
    embedder, store, generator = load_components()

    results = store.search(
        query_embedding=embedder.embed_query(query),
        query_text=query,
        top_k=5,
    )

    return generator.answer(
        query=query,
        results=results,
    )


st.title("Australian Visa RAG Assistant")
st.caption(
    "Version-aware answers grounded in the indexed Australian visa corpus."
)

with st.sidebar:
    st.header("Knowledge Base")

    st.markdown(
        """
**Primary sources**

- Federal Register of Legislation
- Migration Regulations 1994
- Migration Act 1958
- Australian Department of Home Affairs Visa List

**Current indexed corpus**

- 1,618 main chunks
- 320 evidence chunks
- 1,938 total vectors
- C288 consolidated legislation
- Current Home Affairs visa-list snapshot
        """
    )

    st.divider()

    st.markdown(
        """
**Important**

This system provides information from its indexed sources.
It does not replace professional migration advice.
        """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input(
    "Ask about an Australian visa, subclass, requirement, or rule..."
)

if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching authoritative sources..."):
            try:
                answer = answer_question(query)
                st.markdown(answer)
            except Exception as exc:
                answer = (
                    "The chatbot could not generate an answer right now.\n\n"
                    f"Error: `{exc}`"
                )
                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
