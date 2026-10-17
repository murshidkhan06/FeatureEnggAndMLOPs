"""app/services/sessions.py

In-memory conversation memory. It exists so follow-ups like "what if attendance were 95?"
can reuse values the user already gave.

Honest limits (fine for a classroom, NOT for production): it lives in this process's RAM,
vanishes on restart, and only works with a single worker. A real system would use Redis or
a database, with retention rules for student data.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Conversation:
    id: str
    messages: list[dict] = field(default_factory=list)   # OpenAI-style messages (no system prompt)
    scores_seen: list[float] = field(default_factory=list)  # real model outputs seen so far
    touched_at: float = field(default_factory=time.monotonic)

    def user_text(self, extra: str = "") -> str:
        parts = [m["content"] for m in self.messages if m.get("role") == "user" and m.get("content")]
        if extra:
            parts.append(extra)
        return "\n".join(parts)


class ConversationStore:
    def __init__(self, ttl_seconds: int, max_conversations: int, max_messages: int):
        self._items: OrderedDict[str, Conversation] = OrderedDict()
        self._ttl = ttl_seconds
        self._max = max_conversations
        self._max_messages = max_messages

    def get_or_create(self, conversation_id: str | None) -> Conversation:
        self._evict()
        if conversation_id and conversation_id in self._items:
            conv = self._items[conversation_id]
            conv.touched_at = time.monotonic()
            self._items.move_to_end(conversation_id)
            return conv
        return Conversation(id=conversation_id or uuid.uuid4().hex)

    def save(self, conv: Conversation) -> None:
        conv.messages = self._trim(conv.messages)
        conv.touched_at = time.monotonic()
        self._items[conv.id] = conv
        self._items.move_to_end(conv.id)
        self._evict()

    def delete(self, conversation_id: str) -> bool:
        return self._items.pop(conversation_id, None) is not None

    def __len__(self) -> int:
        return len(self._items)

    # -- internals --------------------------------------------------------------------
    def _evict(self) -> None:
        now = time.monotonic()
        for key in [k for k, c in self._items.items() if now - c.touched_at > self._ttl]:
            del self._items[key]
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def _trim(self, messages: list[dict]) -> list[dict]:
        """Keep the newest messages, but always start at a user message so we never keep a
        'tool' result whose matching tool-call was cut off (providers reject that)."""
        if len(messages) <= self._max_messages:
            return messages
        start = len(messages) - self._max_messages
        while start < len(messages) and messages[start].get("role") != "user":
            start += 1
        return messages[start:]
