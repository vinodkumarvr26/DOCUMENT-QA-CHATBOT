"""Offline tests for conversational and document-grounded prompt construction."""

from types import SimpleNamespace
from unittest.mock import Mock

from chatbot.config import Settings
from chatbot.conversation_memory import ConversationTurn
from chatbot.rag_integration import DocumentChunk, RetrievedChunk
from chatbot.stream_handler import (
    stream_chat_response,
    stream_document_response,
)


def make_settings() -> Settings:
    return Settings(
        gemini_api_key="synthetic-key",
        gemini_chat_model="configured-model",
        gemini_embedding_model="configured-embedding-model",
        gemini_embedding_dimensions=3,
        chat_history_limit=2,
        chunk_size=1000,
        chunk_overlap=150,
        retriever_top_k=4,
    )


def test_normal_chat_prompt_includes_history_and_current_question(monkeypatch) -> None:
    client = Mock()
    client.models.generate_content_stream.return_value = iter(
        [SimpleNamespace(text="answer")]
    )
    turns = [ConversationTurn("earlier", "reply")]

    chunks = list(stream_chat_response(client, make_settings(), turns, "current"))

    assert chunks == ["answer"]
    call = client.models.generate_content_stream.call_args.kwargs
    assert [item.parts[0].text for item in call["contents"]] == [
        "earlier",
        "reply",
        "current",
    ]


def test_document_prompt_contains_retrieved_context_and_citation_label() -> None:
    client = Mock()
    client.models.generate_content_stream.return_value = iter(
        [SimpleNamespace(text="Grounded answer")]
    )
    result = RetrievedChunk(
        DocumentChunk(
            "The agreement starts in March.",
            "agreement.pdf",
            4,
            0,
            (1.0, 0.0, 0.0),
        ),
        0.92,
    )

    chunks = list(
        stream_document_response(
            client,
            make_settings(),
            "When does it start?",
            [result],
        )
    )

    assert chunks == ["Grounded answer"]
    call = client.models.generate_content_stream.call_args.kwargs
    prompt = call["contents"][0].parts[0].text
    assert "The agreement starts in March." in prompt
    assert "[Source: agreement.pdf, page 4]" in prompt
    assert "When does it start?" in prompt
    assert "say so clearly" in prompt
    assert call["config"].system_instruction
