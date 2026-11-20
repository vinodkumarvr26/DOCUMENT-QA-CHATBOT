"""Google Gen AI conversation turns and bounded history formatting."""

from dataclasses import dataclass
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any
from uuid import uuid4

from google.genai import types


SESSION_MESSAGES_KEY = "messages"
SESSION_ID_KEY = "conversation_id"


@dataclass(frozen=True)
class ConversationTurn:
    """A completed user/assistant exchange, without client configuration."""

    user_text: str
    assistant_text: str


def initialize_conversation_state(state: MutableMapping[str, Any]) -> None:
    """Create isolated conversation state for a Streamlit browser session."""
    if SESSION_ID_KEY not in state:
        state[SESSION_ID_KEY] = uuid4().hex
    if SESSION_MESSAGES_KEY not in state:
        state[SESSION_MESSAGES_KEY] = []


def start_new_conversation(state: MutableMapping[str, Any]) -> None:
    """Clear the active transcript and assign a fresh conversation identifier."""
    state[SESSION_ID_KEY] = uuid4().hex
    state[SESSION_MESSAGES_KEY] = []


def clear_conversation(state: MutableMapping[str, Any]) -> None:
    """Clear all display and context messages from the active conversation."""
    state[SESSION_MESSAGES_KEY] = []


def enforce_history_limit(
    state: MutableMapping[str, Any], history_limit: int
) -> None:
    """Keep only the newest configured number of displayed conversation turns."""
    if history_limit < 1:
        raise ValueError("history_limit must be at least 1")

    messages = state[SESSION_MESSAGES_KEY]
    ordered_turn_ids = list(dict.fromkeys(message["turn_id"] for message in messages))
    retained_turn_ids = set(ordered_turn_ids[-history_limit:])
    state[SESSION_MESSAGES_KEY] = [
        message for message in messages if message["turn_id"] in retained_turn_ids
    ]


def append_user_message(
    state: MutableMapping[str, Any], content: str
) -> str | None:
    """Store a non-empty user message and return its turn identifier."""
    text = content.strip()
    if not text:
        return None

    turn_id = uuid4().hex
    state[SESSION_MESSAGES_KEY].append(
        {
            "turn_id": turn_id,
            "role": "user",
            "content": text,
            "include_in_context": True,
            "is_error": False,
        }
    )
    return turn_id


def append_assistant_message(
    state: MutableMapping[str, Any],
    turn_id: str,
    content: str,
    *,
    include_in_context: bool = True,
    is_error: bool = False,
    citations: Sequence[str] = (),
) -> None:
    """Store an assistant response, optionally excluding it from future context."""
    state[SESSION_MESSAGES_KEY].append(
        {
            "turn_id": turn_id,
            "role": "assistant",
            "content": content,
            "include_in_context": include_in_context,
            "is_error": is_error,
            "citations": list(citations),
        }
    )


def completed_turns_from_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[ConversationTurn]:
    """Build ordered turns from successful user/assistant message pairs only."""
    turns: dict[str, dict[str, str]] = {}
    ordered_ids: list[str] = []

    for message in messages:
        if not message.get("include_in_context", False):
            continue

        turn_id = message["turn_id"]
        role = message["role"]
        if role not in ("user", "assistant"):
            continue

        if turn_id not in turns:
            turns[turn_id] = {}
            ordered_ids.append(turn_id)
        turns[turn_id][role] = message["content"]

    return [
        ConversationTurn(
            user_text=turns[turn_id]["user"],
            assistant_text=turns[turn_id]["assistant"],
        )
        for turn_id in ordered_ids
        if "user" in turns[turn_id] and "assistant" in turns[turn_id]
    ]


def get_completed_turns(state: MutableMapping[str, Any]) -> list[ConversationTurn]:
    """Return only successful, complete turns in their original order."""
    return completed_turns_from_messages(state[SESSION_MESSAGES_KEY])


def build_conversation_history(
    turns: Sequence[ConversationTurn], history_limit: int
) -> list[types.Content]:
    """Format the newest turns as ordered Google Gen AI content objects."""
    if history_limit < 1:
        raise ValueError("history_limit must be at least 1")

    history: list[types.Content] = []
    for turn in turns[-history_limit:]:
        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=turn.user_text)],
            )
        )
        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=turn.assistant_text)],
            )
        )
    return history
