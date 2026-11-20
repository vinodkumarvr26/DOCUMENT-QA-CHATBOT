"""Offline tests for direct Google Gen AI client and streaming helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import httpx2
from google.genai import errors, types

from chatbot.chat_engine import (
    GeminiConfigurationError,
    GeminiRequestError,
    create_gemini_client,
    stream_gemini_response,
)
from chatbot.config import Settings
from chatbot.conversation_memory import ConversationTurn, build_conversation_history


def make_settings(api_key: str | None) -> Settings:
    return Settings(
        gemini_api_key=api_key,
        gemini_chat_model="test-model",
        gemini_embedding_model="test-embedding-model",
        gemini_embedding_dimensions=768,
        chat_history_limit=5,
        chunk_size=1000,
        chunk_overlap=150,
        retriever_top_k=4,
    )


def test_client_initialization_uses_configured_key_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr("chatbot.chat_engine.genai.Client", factory)

    result = create_gemini_client(make_settings("synthetic-test-key"))

    assert result is client
    factory.assert_called_once_with(api_key="synthetic-test-key")


def test_client_initialization_requires_an_api_key() -> None:
    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        create_gemini_client(make_settings(None))


def test_streaming_yields_non_empty_text_and_uses_separate_system_prompt() -> None:
    user_content = types.Content(
        role="user", parts=[types.Part.from_text(text="Question")]
    )
    chunks = [SimpleNamespace(text="Hello"), SimpleNamespace(text=""), SimpleNamespace(text=" world")]
    client = Mock()
    client.models.generate_content_stream.return_value = iter(chunks)

    result = list(
        stream_gemini_response(
            client,
            "test-model",
            [user_content],
            "Answer briefly.",
        )
    )

    assert result == ["Hello", " world"]
    call = client.models.generate_content_stream.call_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["contents"] == [user_content]
    assert call.kwargs["config"].system_instruction == "Answer briefly."


def test_streaming_handles_empty_response() -> None:
    client = Mock()
    client.models.generate_content_stream.return_value = iter(
        [SimpleNamespace(text=None), SimpleNamespace(text="")]
    )

    assert list(stream_gemini_response(client, "test-model", [], "system")) == []


def test_streaming_wraps_api_errors_without_exposing_response_details() -> None:
    client = Mock()
    client.models.generate_content_stream.side_effect = errors.ClientError(
        429, {"error": {"message": "private provider detail"}}
    )

    with pytest.raises(GeminiRequestError, match="HTTP 429") as error:
        list(stream_gemini_response(client, "test-model", [], "system"))

    assert "private provider detail" not in str(error.value)


def test_streaming_wraps_network_errors_without_exposing_details() -> None:
    client = Mock()
    client.models.generate_content_stream.side_effect = httpx2.ConnectError(
        "private network detail"
    )

    with pytest.raises(GeminiRequestError, match="could not reach the service") as error:
        list(stream_gemini_response(client, "test-model", [], "system"))

    assert "private network detail" not in str(error.value)


def test_streaming_wraps_unexpected_errors_without_exposing_details() -> None:
    client = Mock()
    client.models.generate_content_stream.side_effect = RuntimeError(
        "private SDK detail"
    )

    with pytest.raises(GeminiRequestError) as error:
        list(stream_gemini_response(client, "test-model", [], "system"))

    assert error.value.category == "unexpected"
    assert "private SDK detail" not in str(error.value)


def test_history_builder_does_not_add_client_credentials() -> None:
    turns = [ConversationTurn("question", "answer")]

    history = build_conversation_history(turns, history_limit=3)

    assert [content.role for content in history] == ["user", "model"]
    assert [content.parts[0].text for content in history] == ["question", "answer"]
    assert "synthetic-test-key" not in str(history)
