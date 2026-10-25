
"""Bounded, thread-safe in-memory conversation history and jurisdiction."""

from collections import OrderedDict
from threading import RLock


class ConversationStore:
    """Store recent messages and jurisdiction separately per conversation."""

    def __init__(
        self,
        max_messages: int = 12,
        max_conversations: int = 100,
    ) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")
        if max_conversations < 1:
            raise ValueError("max_conversations must be at least 1.")

        self.max_messages = max_messages
        self.max_conversations = max_conversations
        self._conversations: OrderedDict[str, list[dict[str, str]]] = (
            OrderedDict()
        )
        self._jurisdictions: dict[str, str] = {}
        self._lock = RLock()

    def _touch(self, conversation_id: str) -> None:
        self._conversations.move_to_end(conversation_id)

    def _enforce_limit(self) -> None:
        while len(self._conversations) > self.max_conversations:
            oldest_id, _ = self._conversations.popitem(last=False)
            self._jurisdictions.pop(oldest_id, None)

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Return a copy of recent history, or an empty list."""
        with self._lock:
            history = self._conversations.get(conversation_id, [])
            if conversation_id in self._conversations:
                self._touch(conversation_id)
            return [message.copy() for message in history]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append one user or assistant message and enforce storage limits."""
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string.")

        message = {"role": role, "content": content[:4000]}

        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []

            history = self._conversations[conversation_id]
            history.append(message)
            del history[:-self.max_messages]

            self._touch(conversation_id)
            self._enforce_limit()

    def get_jurisdiction(self, conversation_id: str) -> str | None:
        """Return the last selected destination code for this conversation."""
        with self._lock:
            if conversation_id not in self._conversations:
                return None
            self._touch(conversation_id)
            return self._jurisdictions.get(conversation_id)

    def set_jurisdiction(
        self,
        conversation_id: str,
        jurisdiction: str,
    ) -> None:
        """Store a destination code separately from message text."""
        if not isinstance(jurisdiction, str) or not jurisdiction.strip():
            raise ValueError("jurisdiction must be a non-empty string.")

        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []

            self._jurisdictions[conversation_id] = jurisdiction
            self._touch(conversation_id)
            self._enforce_limit()

    def clear(self, conversation_id: str) -> None:
        """Remove one conversation and its jurisdiction."""
        with self._lock:
            self._conversations.pop(conversation_id, None)
            self._jurisdictions.pop(conversation_id, None)
