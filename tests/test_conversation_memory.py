"""Offline tests for bounded chronological conversation history."""

import pytest

from chatbot.conversation_memory import (
    SESSION_ID_KEY,
    SESSION_MESSAGES_KEY,
    ConversationTurn,
    append_assistant_message,
    append_user_message,
    build_conversation_history,
    clear_conversation,
    enforce_history_limit,
    get_completed_turns,
    initialize_conversation_state,
    start_new_conversation,
)


def test_history_preserves_turn_order() -> None:
    turns = [
        ConversationTurn("first question", "first answer"),
        ConversationTurn("second question", "second answer"),
    ]

    history = build_conversation_history(turns, history_limit=5)

    assert [(item.role, item.parts[0].text) for item in history] == [
        ("user", "first question"),
        ("model", "first answer"),
        ("user", "second question"),
        ("model", "second answer"),
    ]


def test_history_limit_keeps_newest_completed_turns() -> None:
    turns = [
        ConversationTurn("first question", "first answer"),
        ConversationTurn("second question", "second answer"),
        ConversationTurn("third question", "third answer"),
    ]

    history = build_conversation_history(turns, history_limit=2)

    assert [item.parts[0].text for item in history] == [
        "second question",
        "second answer",
        "third question",
        "third answer",
    ]


def test_history_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_conversation_history([], history_limit=0)


def test_session_states_are_isolated() -> None:
    first_session: dict[str, object] = {}
    second_session: dict[str, object] = {}

    initialize_conversation_state(first_session)
    initialize_conversation_state(second_session)
    append_user_message(first_session, "private question")

    assert first_session[SESSION_ID_KEY] != second_session[SESSION_ID_KEY]
    assert len(first_session[SESSION_MESSAGES_KEY]) == 1
    assert second_session[SESSION_MESSAGES_KEY] == []


def test_clear_conversation_removes_display_and_context_history() -> None:
    state: dict[str, object] = {}
    initialize_conversation_state(state)
    turn_id = append_user_message(state, "question")
    assert turn_id is not None
    append_assistant_message(state, turn_id, "answer")

    clear_conversation(state)

    assert state[SESSION_MESSAGES_KEY] == []
    assert get_completed_turns(state) == []


def test_new_conversation_resets_messages_and_rotates_id() -> None:
    state: dict[str, object] = {}
    initialize_conversation_state(state)
    original_id = state[SESSION_ID_KEY]
    append_user_message(state, "question")

    start_new_conversation(state)

    assert state[SESSION_ID_KEY] != original_id
    assert state[SESSION_MESSAGES_KEY] == []


def test_empty_user_messages_are_not_stored() -> None:
    state: dict[str, object] = {}
    initialize_conversation_state(state)

    assert append_user_message(state, "  \n ") is None
    assert state[SESSION_MESSAGES_KEY] == []


def test_failed_assistant_response_is_not_added_to_gemini_context() -> None:
    state: dict[str, object] = {}
    initialize_conversation_state(state)
    turn_id = append_user_message(state, "failed question")
    assert turn_id is not None
    append_assistant_message(
        state,
        turn_id,
        "safe error shown to user",
        include_in_context=False,
        is_error=True,
    )

    assert get_completed_turns(state) == []


def test_session_transcript_is_bounded_to_configured_turn_limit() -> None:
    state: dict[str, object] = {}
    initialize_conversation_state(state)
    for index in range(3):
        turn_id = append_user_message(state, f"question {index}")
        assert turn_id is not None
        append_assistant_message(state, turn_id, f"answer {index}")

    enforce_history_limit(state, history_limit=2)

    assert [
        message["content"] for message in state[SESSION_MESSAGES_KEY]
    ] == ["question 1", "answer 1", "question 2", "answer 2"]
