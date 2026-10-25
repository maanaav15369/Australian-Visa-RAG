from types import SimpleNamespace

from app.answer_generator import AnswerGenerator


class FakeCohereClient:
    def __init__(self):
        self.messages = None

    def chat(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(
            message=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text="Here is the information supported by the source [S1].",
                    )
                ]
            )
        )


def test_answer_prompt_uses_counsellor_style_and_preserves_citations():
    generator = AnswerGenerator.__new__(AnswerGenerator)
    fake_client = FakeCohereClient()
    generator.client = fake_client

    results = {
        "ids": [["chunk-1"]],
        "documents": [["Official evidence about a visa requirement."]],
        "metadatas": [[
            {
                "title": "Test official source",
                "authority": "Test authority",
                "status": "CURRENT",
                "version": "v1",
                "line_start": 1,
                "line_end": 10,
                "source_url": "https://example.org/source",
            }
        ]],
    }

    answer = generator.answer(
        query="I want to understand my immigration options. Where should I start?",
        results=results,
    )

    assert "[S1]" in answer
    assert fake_client.messages is not None
    prompt = fake_client.messages[1]["content"]

    assert "counsellor-style voice" in prompt
    assert "plain English" in prompt
    assert "at most one focused follow-up question" in prompt
    assert "Every substantive legal claim must include the relevant source marker" in prompt
    assert "Never invent a URL or source detail" in prompt


def test_answer_generator_includes_history_as_context_not_evidence():
    generator = AnswerGenerator.__new__(AnswerGenerator)
    fake_client = FakeCohereClient()
    generator.client = fake_client

    results = {
        "ids": [["chunk-1"]],
        "documents": [["Official evidence."]],
        "metadatas": [[{
            "title": "Official source",
            "status": "CURRENT",
            "version": "v1",
            "line_start": 1,
            "line_end": 2,
        }]],
    }

    generator.answer(
        query="What should I check next?",
        results=results,
        history=[
            {"role": "user", "content": "I want to study in Australia."},
            {"role": "assistant", "content": "Which level of study?"},
        ],
    )

    prompt = fake_client.messages[1]["content"]
    assert "RECENT CONVERSATION" in prompt
    assert "I want to study in Australia." in prompt
    assert "never treat it as legal evidence" in prompt
    assert "RETRIEVED AUTHORITATIVE CONTEXT" in prompt
