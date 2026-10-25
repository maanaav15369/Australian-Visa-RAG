import pytest

from visa_intel.conversation_store import ConversationStore


def test_conversations_are_isolated():
    store = ConversationStore()

    store.add_message("conversation_a", "user", "Australia study")
    store.add_message("conversation_b", "user", "Canada study")

    assert store.get_history("conversation_a") == [
        {"role": "user", "content": "Australia study"}
    ]
    assert store.get_history("conversation_b") == [
        {"role": "user", "content": "Canada study"}
    ]


def test_history_keeps_only_most_recent_messages():
    store = ConversationStore(max_messages=2)

    store.add_message("conversation_a", "user", "first")
    store.add_message("conversation_a", "assistant", "second")
    store.add_message("conversation_a", "user", "third")

    assert store.get_history("conversation_a") == [
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]


def test_returned_history_cannot_mutate_stored_history():
    store = ConversationStore()
    store.add_message("conversation_a", "user", "original")

    history = store.get_history("conversation_a")
    history[0]["content"] = "changed"
    history.append({"role": "assistant", "content": "injected"})

    assert store.get_history("conversation_a") == [
        {"role": "user", "content": "original"}
    ]


def test_oldest_conversation_is_evicted_at_capacity():
    store = ConversationStore(max_conversations=2)

    store.add_message("first_chat", "user", "first")
    store.add_message("second_chat", "user", "second")
    store.add_message("third_chat", "user", "third")

    assert store.get_history("first_chat") == []
    assert store.get_history("second_chat")
    assert store.get_history("third_chat")


def test_clear_removes_only_selected_conversation():
    store = ConversationStore()
    store.add_message("conversation_a", "user", "A")
    store.add_message("conversation_b", "user", "B")

    store.clear("conversation_a")

    assert store.get_history("conversation_a") == []
    assert store.get_history("conversation_b") == [
        {"role": "user", "content": "B"}
    ]


@pytest.mark.parametrize("role", ["system", "tool", "developer"])
def test_invalid_roles_are_rejected(role):
    store = ConversationStore()

    with pytest.raises(ValueError):
        store.add_message("conversation_a", role, "content")


def test_message_content_is_bounded():
    store = ConversationStore()
    store.add_message("conversation_a", "user", "x" * 5000)

    assert len(store.get_history("conversation_a")[0]["content"]) == 4000
