"""Streamlit-level tests for user submission and streamed assistant replies."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

from streamlit.testing.v1 import AppTest

from chatbot import chat_engine, config, stream_handler
from chatbot.config import Settings
from chatbot.conversation_memory import ConversationTurn
from chatbot import rag_integration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"


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


def test_chat_submission_streams_response_and_uses_follow_up_context(
    monkeypatch,
) -> None:
    settings = make_settings("synthetic-test-key")
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    client = MagicMock()
    monkeypatch.setattr(
        chat_engine, "create_gemini_client", lambda configured: client
    )

    received_turns: list[list[ConversationTurn]] = []

    def fake_stream(_client, _settings, turns, question):
        received_turns.append(list(turns))
        assert question in ("First question", "Follow-up question")
        return iter(["A streamed ", "answer."])

    monkeypatch.setattr(stream_handler, "stream_chat_response", fake_stream)

    app = AppTest.from_file(str(APP_PATH)).run()
    app.chat_input[0].set_value("First question").run()

    assert not app.exception
    assert [
        (message["role"], message["content"])
        for message in app.session_state["messages"]
    ] == [
        ("user", "First question"),
        ("assistant", "A streamed answer."),
    ]

    app.chat_input[0].set_value("Follow-up question").run()

    assert received_turns[0] == []
    assert received_turns[1] == [
        ConversationTurn("First question", "A streamed answer.")
    ]
    assert [
        message["content"] for message in app.session_state["messages"]
    ] == [
        "First question",
        "A streamed answer.",
        "Follow-up question",
        "A streamed answer.",
    ]

    app.button(key="clear_conversation").click().run()
    assert app.session_state["messages"] == []


def test_document_chat_retrieves_sources_and_displays_citations(
    monkeypatch,
) -> None:
    from chatbot.rag_integration import DocumentChunk, RetrievedChunk

    settings = make_settings("synthetic-test-key")
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_engine, "create_gemini_client", lambda _settings: MagicMock()
    )
    document_chunk = DocumentChunk(
        "The policy allows ten days.",
        "policy.txt",
        None,
        0,
        (1.0, 0.0, 0.0),
    )
    retrieved = [RetrievedChunk(document_chunk, 0.95)]
    retrieval_questions: list[str] = []
    monkeypatch.setattr(
        rag_integration,
        "retrieve_relevant_chunks",
        lambda _client, _settings, question, _documents: (
            retrieval_questions.append(question) or retrieved
        ),
    )
    document_questions: list[str] = []

    def fake_document_stream(_client, _settings, question, chunks):
        document_questions.append(question)
        assert chunks == retrieved
        return iter(["The policy allows ten days."])

    monkeypatch.setattr(
        stream_handler, "stream_document_response", fake_document_stream
    )
    normal_stream = Mock(side_effect=AssertionError("normal chat should not run"))
    monkeypatch.setattr(stream_handler, "stream_chat_response", normal_stream)

    app = AppTest.from_file(str(APP_PATH)).run()
    rag_integration.initialize_document_state(app.session_state)
    document = rag_integration.IndexedDocument("policy-id", "policy.txt", ())
    rag_integration.add_document_to_session(app.session_state, document)
    app.run()
    app.chat_input[0].set_value("How long is the policy?").run()

    assert not app.exception
    assert retrieval_questions == ["How long is the policy?"]
    assert document_questions == ["How long is the policy?"]
    normal_stream.assert_not_called()
    assistant_message = app.session_state["messages"][-1]
    assert assistant_message["citations"] == ["policy.txt"]


def test_missing_api_key_disables_chat_input(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_settings", lambda: make_settings(None))
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.button(key="index_documents").disabled
    assert any("GEMINI_API_KEY" in info.value for info in app.info)


def test_api_error_is_shown_safely_and_excluded_from_context(monkeypatch) -> None:
    from chatbot.chat_engine import GeminiRequestError

    settings = make_settings("synthetic-test-key")
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_engine, "create_gemini_client", lambda _settings: MagicMock()
    )

    def failed_stream(_client, _settings, _turns, _question):
        raise GeminiRequestError(
            "provider response contains private details", status_code=429
        )
        yield ""

    monkeypatch.setattr(stream_handler, "stream_chat_response", failed_stream)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.chat_input[0].set_value("Please answer").run()

    assert not app.exception
    messages = app.session_state["messages"]
    assert messages[-1]["is_error"] is True
    assert "quota or rate limit" in messages[-1]["content"]
    assert "private details" not in messages[-1]["content"]


def test_upload_index_remove_and_clear_controls(monkeypatch) -> None:
    settings = make_settings("synthetic-test-key")
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_engine, "create_gemini_client", lambda _settings: MagicMock()
    )

    def fake_index(
        _client,
        _settings,
        filename,
        content,
        max_chunks,
        progress_callback,
    ):
        document_id = rag_integration.document_id_for_upload(filename, content)
        return rag_integration.IndexedDocument(document_id, filename, ())

    monkeypatch.setattr(rag_integration, "index_document", fake_index)

    app = AppTest.from_file(str(APP_PATH)).run()
    uploader = app.file_uploader[0]
    uploader.upload("guide.txt", b"Useful guide text", "text/plain")
    uploader.upload("notes.txt", b"Additional notes", "text/plain")
    uploader.run()
    assert not app.exception

    app.button(key="index_documents").click().run()
    assert not app.exception
    documents = app.session_state[rag_integration.DOCUMENTS_KEY]
    assert {document.filename for document in documents.values()} == {
        "guide.txt",
        "notes.txt",
    }

    guide_id = rag_integration.document_id_for_upload(
        "guide.txt", b"Useful guide text"
    )
    app.button(key=f"remove_document_{guide_id}").click().run()
    assert guide_id not in app.session_state[rag_integration.DOCUMENTS_KEY]
    assert len(app.session_state[rag_integration.DOCUMENTS_KEY]) == 1

    app.button(key="clear_documents").click().run()
    assert app.session_state[rag_integration.DOCUMENTS_KEY] == {}
    assert app.session_state[rag_integration.DOCUMENT_STATUSES_KEY] == {}


def test_embedding_failure_marks_upload_failed_without_partial_index(monkeypatch) -> None:
    settings = make_settings("synthetic-test-key")
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_engine, "create_gemini_client", lambda _settings: MagicMock()
    )

    def failed_index(
        _client,
        _settings,
        _filename,
        _content,
        max_chunks,
        progress_callback,
    ):
        raise rag_integration.EmbeddingError(
            "safe internal error", status_code=429, category="quota"
        )

    monkeypatch.setattr(rag_integration, "index_document", failed_index)
    app = AppTest.from_file(str(APP_PATH)).run()
    upload = app.file_uploader[0]
    upload.upload("guide.txt", b"Useful guide text", "text/plain")
    upload.run()
    app.button(key="index_documents").click().run()

    document_id = rag_integration.document_id_for_upload(
        "guide.txt", b"Useful guide text"
    )
    assert not app.exception
    assert app.session_state[rag_integration.DOCUMENTS_KEY] == {}
    assert (
        app.session_state[rag_integration.DOCUMENT_STATUSES_KEY][document_id][
            "status"
        ]
        == "Failed"
    )
    assert "quota or rate limit" in (
        app.session_state[rag_integration.DOCUMENT_STATUSES_KEY][document_id][
            "message"
        ]
    )
    assert "safe internal error" not in str(app.session_state)


def test_document_upload_count_limit_disables_indexing(monkeypatch) -> None:
    settings = make_settings("synthetic-test-key")
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    app = AppTest.from_file(str(APP_PATH)).run()
    uploader = app.file_uploader[0]
    for index in range(6):
        uploader.upload(
            f"guide-{index}.txt",
            b"Useful guide text",
            "text/plain",
        )
    uploader.run()

    assert not app.exception
    assert app.button(key="index_documents").disabled
    assert any("no more than" in warning.value for warning in app.warning)
