"""Offline tests for document extraction, indexing, and retrieval."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import errors
import httpx2
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from chatbot.config import Settings
from chatbot.rag_integration import (
    DOCUMENTS_KEY,
    DOCUMENT_STATUSES_KEY,
    DocumentChunk,
    DocumentProcessingError,
    EmbeddingError,
    IndexedDocument,
    RetrievedChunk,
    add_document_to_session,
    clear_session_documents,
    document_id_for_upload,
    embed_texts,
    extract_document_pages,
    format_source_citations,
    index_document,
    initialize_document_state,
    remove_document_from_session,
    retrieve_relevant_chunks,
    set_document_status,
    split_text,
)


def make_settings(
    *,
    api_key: str | None = "synthetic-test-key",
    chunk_size: int = 30,
    chunk_overlap: int = 5,
    top_k: int = 2,
    embedding_dimensions: int = 3,
) -> Settings:
    return Settings(
        gemini_api_key=api_key,
        gemini_chat_model="configured-chat-model",
        gemini_embedding_model="configured-embedding-model",
        gemini_embedding_dimensions=embedding_dimensions,
        chat_history_limit=5,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retriever_top_k=top_k,
    )


def make_text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=100)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(
        f"BT /F1 12 Tf 10 50 Td ({escaped_text}) Tj ET".encode("ascii")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def mock_embedding_client(vectors: list[tuple[float, ...]]) -> Mock:
    client = Mock()
    responses = [
        SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=list(vector))
            ]
        )
        for vector in vectors
    ]
    if responses:
        client.models.embed_content.side_effect = responses
    else:
        client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[]
        )
    return client


def test_extracts_pdf_text_and_retains_one_based_page_number() -> None:
    pages = extract_document_pages("notes.pdf", make_text_pdf("PDF source text"))

    assert pages == [(1, "PDF source text")]


def test_extracts_utf8_txt_and_preserves_no_page_metadata() -> None:
    pages = extract_document_pages("notes.txt", b"\xef\xbb\xbfFirst line\nSecond line")

    assert pages == [(None, "First line Second line")]


def test_rejects_empty_unsupported_corrupt_and_scanned_documents() -> None:
    with pytest.raises(DocumentProcessingError, match="empty"):
        extract_document_pages("empty.txt", b"")
    with pytest.raises(DocumentProcessingError, match="Unsupported"):
        extract_document_pages("notes.md", b"text")
    with pytest.raises(DocumentProcessingError, match="UTF-8"):
        extract_document_pages("bad.txt", b"\xff")
    with pytest.raises(DocumentProcessingError, match="corrupt"):
        extract_document_pages("bad.pdf", b"not a pdf")
    with pytest.raises(DocumentProcessingError, match="scanned/image-only"):
        extract_document_pages("scanned.pdf", _blank_pdf())


def test_index_rejects_documents_over_the_chunk_budget_before_embedding() -> None:
    client = Mock()
    settings = make_settings(chunk_size=10, chunk_overlap=1)

    with pytest.raises(DocumentProcessingError, match="in-memory indexing limit"):
        index_document(
            client,
            settings,
            "long.txt",
            b"one two three four five six seven eight nine ten",
            max_chunks=1,
        )

    client.models.embed_content.assert_not_called()


def _blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_split_text_respects_chunk_size_overlap_and_word_boundaries() -> None:
    chunks = split_text("one two three four five six seven", 15, 4)

    assert chunks
    assert all(len(chunk) <= 15 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ").startswith("one two three")
    assert chunks[0].split()[-1] in chunks[1].split()


def test_indexing_preserves_source_page_and_chunk_metadata() -> None:
    settings = make_settings(chunk_size=100, chunk_overlap=5)
    client = mock_embedding_client([(1.0, 0.0, 0.0)])

    indexed = index_document(
        client, settings, "source.pdf", make_text_pdf("PDF text continues here")
    )

    assert indexed.filename == "source.pdf"
    assert indexed.chunks
    assert all(chunk.source_filename == "source.pdf" for chunk in indexed.chunks)
    assert all(chunk.page_number == 1 for chunk in indexed.chunks)
    assert indexed.chunks[0].metadata["page_number"] == 1
    assert indexed.chunks[0].metadata["source_filename"] == "source.pdf"
    client.models.embed_content.assert_called_once()


def test_embedding_uses_configured_model_dimensions_and_batches(monkeypatch) -> None:
    settings = make_settings(embedding_dimensions=3)
    client = mock_embedding_client([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])

    vectors = embed_texts(client, settings, ["first", "second"])

    assert vectors == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    assert client.models.embed_content.call_count == 2
    calls = client.models.embed_content.call_args_list
    assert [call.kwargs["contents"] for call in calls] == ["first", "second"]
    assert all(
        call.kwargs["model"] == "configured-embedding-model"
        and call.kwargs["config"].output_dimensionality == 3
        and call.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"
        for call in calls
    )


def test_embedding_reports_progress_after_each_completed_chunk() -> None:
    client = mock_embedding_client([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    progress: list[tuple[int, int]] = []

    embed_texts(
        client,
        make_settings(),
        ["first", "second"],
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert progress == [(1, 2), (2, 2)]


def test_embedding_api_failure_is_wrapped_without_provider_details() -> None:
    client = Mock()
    client.models.embed_content.side_effect = errors.ClientError(
        429, {"error": {"message": "private provider detail"}}
    )

    with pytest.raises(EmbeddingError, match="request failed") as error:
        embed_texts(client, make_settings(), ["text"])

    assert "private provider detail" not in str(error.value)
    assert error.value.status_code == 429
    assert error.value.category == "quota"


@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (404, "unsupported_model"),
        (400, "invalid_request"),
        (429, "quota"),
        (503, "service"),
    ],
)
def test_embedding_api_errors_are_classified_safely(
    status_code: int, category: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = Mock()
    client.models.embed_content.side_effect = errors.ClientError(
        status_code, {"error": {"message": "private provider detail"}}
    )

    with pytest.raises(EmbeddingError) as error:
        embed_texts(client, make_settings(), ["private document text"])

    assert error.value.status_code == status_code
    assert error.value.category == category
    assert "private provider detail" not in caplog.text
    assert "private document text" not in caplog.text


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (httpx2.ConnectError("private connection detail"), "network"),
        (httpx2.ReadTimeout("private timeout detail"), "timeout"),
        (RuntimeError("private SDK detail"), "unexpected"),
    ],
)
def test_embedding_transport_and_unexpected_failures_are_safe(
    failure: Exception, category: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = Mock()
    client.models.embed_content.side_effect = failure

    with pytest.raises(EmbeddingError) as error:
        embed_texts(client, make_settings(), ["private document text"])

    assert error.value.category == category
    assert "private" not in str(error.value)
    assert "private" not in caplog.text


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [(1.0, 0.0)],
    ],
)
def test_embedding_rejects_empty_or_wrong_dimension_response(
    vectors: list[tuple[float, ...]]
) -> None:
    client = mock_embedding_client(vectors)

    with pytest.raises(EmbeddingError) as error:
        embed_texts(client, make_settings(), ["text"])

    assert error.value.category == "invalid_response"


def test_retrieval_uses_cosine_similarity_and_configured_top_k() -> None:
    settings = make_settings(top_k=2)
    chunks = [
        DocumentChunk("aligned", "a.txt", None, 0, (1.0, 0.0, 0.0)),
        DocumentChunk("diagonal", "b.txt", None, 0, (0.8, 0.6, 0.0)),
        DocumentChunk("unrelated", "c.txt", None, 0, (0.0, 1.0, 0.0)),
    ]
    documents = {
        str(index): IndexedDocument(str(index), chunk.source_filename, (chunk,))
        for index, chunk in enumerate(chunks)
    }
    client = mock_embedding_client([(1.0, 0.0, 0.0)])

    results = retrieve_relevant_chunks(
        client, settings, "question", documents
    )

    assert len(results) == 2
    assert [result.chunk.text for result in results] == ["aligned", "diagonal"]
    assert results[0].similarity == pytest.approx(1.0)
    assert (
        client.models.embed_content.call_args.kwargs["config"].task_type
        == "RETRIEVAL_QUERY"
    )


def test_citations_deduplicate_filename_and_page_in_result_order() -> None:
    results = [
        RetrievedChunk(DocumentChunk("a", "guide.pdf", 1, 0, (1.0,)), 0.9),
        RetrievedChunk(DocumentChunk("b", "guide.pdf", 1, 1, (1.0,)), 0.8),
        RetrievedChunk(DocumentChunk("c", "guide.pdf", 2, 2, (1.0,)), 0.7),
        RetrievedChunk(DocumentChunk("d", "notes.txt", None, 0, (1.0,)), 0.6),
    ]

    assert format_source_citations(results) == [
        "guide.pdf, page 1",
        "guide.pdf, page 2",
        "notes.txt",
    ]


def test_document_indexes_are_isolated_and_removable_per_session() -> None:
    first: dict[str, object] = {}
    second: dict[str, object] = {}
    initialize_document_state(first)
    initialize_document_state(second)
    document = IndexedDocument("id", "notes.txt", ())
    add_document_to_session(first, document)

    assert document_id_for_upload("notes.txt", b"x") != document_id_for_upload(
        "other.txt", b"x"
    )
    assert "id" in first[DOCUMENTS_KEY]
    assert second[DOCUMENTS_KEY] == {}

    remove_document_from_session(first, "id")
    assert first[DOCUMENTS_KEY] == {}
    assert first[DOCUMENT_STATUSES_KEY] == {}


def test_clear_removes_all_documents_and_statuses() -> None:
    state: dict[str, object] = {}
    initialize_document_state(state)
    add_document_to_session(state, IndexedDocument("id", "notes.txt", ()))
    set_document_status(state, "failed", "broken.pdf", "Failed", "bad file")

    clear_session_documents(state)

    assert state[DOCUMENTS_KEY] == {}
    assert state[DOCUMENT_STATUSES_KEY] == {}


def test_rag_indexing_rejects_missing_api_key_before_embedding() -> None:
    from chatbot.chat_engine import GeminiConfigurationError, create_gemini_client

    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        create_gemini_client(make_settings(api_key=None))
