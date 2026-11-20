"""Session-scoped PDF and text document retrieval with Google embeddings."""

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import logging
import math
import re
from typing import Any, Callable, Literal

from google import genai
from google.genai import errors, types
import httpx2
from pypdf import PdfReader, errors as pdf_errors

from chatbot.config import Settings


DOCUMENTS_KEY = "rag_documents"
DOCUMENT_STATUSES_KEY = "rag_document_statuses"
MAX_CHUNKS_PER_DOCUMENT = 250
MAX_INDEXED_CHUNKS_PER_SESSION = 500
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
logger = logging.getLogger(__name__)


class DocumentProcessingError(ValueError):
    """Raised when a document cannot be read or chunked safely."""


class EmbeddingError(RuntimeError):
    """Raised when Gemini cannot create valid document embeddings."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        category: str = "invalid_response",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.category = category


@dataclass(frozen=True)
class DocumentChunk:
    """A normalized text chunk with source metadata and an embedding vector."""

    text: str
    source_filename: str
    page_number: int | None
    chunk_index: int
    embedding: tuple[float, ...]

    @property
    def metadata(self) -> dict[str, str | int]:
        """Expose provenance fields for retrieval results and citations."""
        metadata: dict[str, str | int] = {
            "source_filename": self.source_filename,
            "chunk_index": self.chunk_index,
        }
        if self.page_number is not None:
            metadata["page_number"] = self.page_number
        return metadata


@dataclass(frozen=True)
class IndexedDocument:
    """An indexed upload stored only in its owning Streamlit session."""

    document_id: str
    filename: str
    chunks: tuple[DocumentChunk, ...]


@dataclass(frozen=True)
class RetrievedChunk:
    """A relevant chunk with its cosine similarity score."""

    chunk: DocumentChunk
    similarity: float


def initialize_document_state(state: MutableMapping[str, Any]) -> None:
    """Initialize document storage and upload statuses in a session."""
    if DOCUMENTS_KEY not in state:
        state[DOCUMENTS_KEY] = {}
    if DOCUMENT_STATUSES_KEY not in state:
        state[DOCUMENT_STATUSES_KEY] = {}


def document_id_for_upload(filename: str, file_bytes: bytes) -> str:
    """Return a stable session index key for identical filename/content uploads."""
    digest = sha256(file_bytes).hexdigest()
    return sha256(f"{filename}\0{digest}".encode("utf-8")).hexdigest()


def extract_document_pages(
    filename: str, file_bytes: bytes
) -> list[tuple[int | None, str]]:
    """Extract page-level PDF text or UTF-8 text, rejecting empty/scanned input."""
    suffix = filename.lower().rsplit(".", maxsplit=1)
    extension = f".{suffix[-1]}" if len(suffix) == 2 else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(
            "Unsupported document type. Upload a PDF or TXT file."
        )
    if not file_bytes:
        raise DocumentProcessingError("The uploaded document is empty.")

    if extension == ".txt":
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise DocumentProcessingError(
                "The TXT file must use UTF-8 text encoding."
            ) from None
        cleaned = clean_text(text)
        if not cleaned:
            raise DocumentProcessingError("The TXT file contains no readable text.")
        return [(None, cleaned)]

    try:
        reader = PdfReader(BytesIO(file_bytes), strict=True)
        pages = [
            (page_number, clean_text(page.extract_text() or ""))
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    except (
        pdf_errors.PdfReadError,
        pdf_errors.EmptyFileError,
        pdf_errors.FileNotDecryptedError,
        OSError,
        ValueError,
    ):
        raise DocumentProcessingError(
            "The PDF could not be read. It may be corrupt or password-protected."
        ) from None

    if not any(text for _, text in pages):
        raise DocumentProcessingError(
            "No selectable text was found. This may be a scanned/image-only PDF; "
            "OCR is not supported."
        )
    return pages


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining readable word and paragraph spacing."""
    return re.sub(r"\s+", " ", text).strip()


def split_text(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split text into bounded overlapping chunks at word boundaries."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than size")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = max(
                cleaned.rfind(" ", start + chunk_size // 2, end),
                cleaned.rfind("\n", start + chunk_size // 2, end),
            )
            if boundary > start:
                end = boundary

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        next_start = max(start + 1, end - chunk_overlap)
        if next_start < len(cleaned) and cleaned[next_start] != " ":
            previous_boundary = cleaned.rfind(" ", start, next_start)
            if previous_boundary >= start:
                next_start = previous_boundary + 1
        start = max(start + 1, next_start)
    return chunks


def embed_texts(
    client: genai.Client,
    settings: Settings,
    texts: Sequence[str],
    task_type: Literal[
        "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"
    ] = "RETRIEVAL_DOCUMENT",
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[tuple[float, ...]]:
    """Generate vectors using the configured Google embedding model."""
    vectors: list[tuple[float, ...]] = []
    for completed, text in enumerate(texts, start=1):
        try:
            response = client.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=settings.gemini_embedding_dimensions,
                    task_type=task_type,
                ),
            )
            batch_embeddings = response.embeddings or []
            if len(batch_embeddings) != 1:
                logger.error(
                    "Gemini returned an unexpected embedding count: "
                    "model=%s dimensions=%d task=%s expected=1 actual=%d",
                    settings.gemini_embedding_model,
                    settings.gemini_embedding_dimensions,
                    task_type,
                    len(batch_embeddings),
                )
                raise EmbeddingError(
                    "Gemini returned an invalid embedding response.",
                    category="invalid_response",
                )
            for embedding in batch_embeddings:
                values = embedding.values or []
                if not values:
                    logger.error(
                        "Gemini returned an empty embedding vector: "
                        "model=%s dimensions=%d task=%s",
                        settings.gemini_embedding_model,
                        settings.gemini_embedding_dimensions,
                        task_type,
                    )
                    raise EmbeddingError(
                        "Gemini returned an empty embedding.",
                        category="invalid_response",
                    )
                if len(values) != settings.gemini_embedding_dimensions:
                    logger.error(
                        "Gemini returned an unexpected vector dimension: "
                        "model=%s expected_dimensions=%d actual_dimensions=%d "
                        "task=%s",
                        settings.gemini_embedding_model,
                        settings.gemini_embedding_dimensions,
                        len(values),
                        task_type,
                    )
                    raise EmbeddingError(
                        "Gemini returned vectors with an unexpected dimension.",
                        category="invalid_response",
                    )
                vectors.append(tuple(float(value) for value in values))
            if progress_callback is not None:
                progress_callback(completed, len(texts))
        except EmbeddingError:
            logger.error(
                "Gemini embedding response validation failed: "
                "model=%s dimensions=%d task=%s category=%s",
                settings.gemini_embedding_model,
                settings.gemini_embedding_dimensions,
                task_type,
                "invalid_response",
            )
            raise
        except errors.APIError as exc:
            category = _api_error_category(exc.code)
            logger.error(
                "Gemini embedding API failure: model=%s dimensions=%d "
                "task=%s exception=%s status=%d category=%s",
                settings.gemini_embedding_model,
                settings.gemini_embedding_dimensions,
                task_type,
                type(exc).__name__,
                exc.code,
                category,
            )
            raise EmbeddingError(
                "Gemini embedding request failed.",
                status_code=exc.code,
                category=category,
            ) from None
        except httpx2.TimeoutException as exc:
            logger.error(
                "Gemini embedding timed out: model=%s dimensions=%d "
                "task=%s exception=%s",
                settings.gemini_embedding_model,
                settings.gemini_embedding_dimensions,
                task_type,
                type(exc).__name__,
            )
            raise EmbeddingError(
                "Gemini embedding request timed out.",
                category="timeout",
            ) from None
        except (httpx2.RequestError, ConnectionError, OSError) as exc:
            logger.error(
                "Gemini embedding connection failure: model=%s dimensions=%d "
                "task=%s exception=%s",
                settings.gemini_embedding_model,
                settings.gemini_embedding_dimensions,
                task_type,
                type(exc).__name__,
            )
            raise EmbeddingError(
                "Could not connect to Gemini for document embeddings.",
                category="network",
            ) from None
        except TimeoutError as exc:
            logger.error(
                "Gemini embedding timed out: model=%s dimensions=%d "
                "task=%s exception=%s",
                settings.gemini_embedding_model,
                settings.gemini_embedding_dimensions,
                task_type,
                type(exc).__name__,
            )
            raise EmbeddingError(
                "Gemini embedding request timed out.",
                category="timeout",
            ) from None
        except Exception as exc:
            logger.error(
                "Unexpected Gemini embedding failure: model=%s dimensions=%d "
                "task=%s exception=%s",
                settings.gemini_embedding_model,
                settings.gemini_embedding_dimensions,
                task_type,
                type(exc).__name__,
            )
            raise EmbeddingError(
                "An unexpected error occurred while creating embeddings.",
                category="unexpected",
            ) from None
    return vectors


def _api_error_category(status_code: int) -> str:
    if status_code in (401, 403):
        return "authentication"
    if status_code == 404:
        return "unsupported_model"
    if status_code == 400:
        return "invalid_request"
    if status_code == 429:
        return "quota"
    if status_code >= 500:
        return "service"
    return "api"


def index_document(
    client: genai.Client,
    settings: Settings,
    filename: str,
    file_bytes: bytes,
    max_chunks: int = MAX_CHUNKS_PER_DOCUMENT,
    progress_callback: Callable[[int, int], None] | None = None,
) -> IndexedDocument:
    """Extract, chunk, and embed an upload before returning an atomic index."""
    if max_chunks < 1:
        raise DocumentProcessingError(
            "The session reached its document index memory limit. "
            "Remove documents before indexing another file."
        )

    pages = extract_document_pages(filename, file_bytes)
    pending_chunks: list[tuple[str, int | None, int]] = []
    chunk_index = 0
    for page_number, text in pages:
        for chunk_text in split_text(
            text, settings.chunk_size, settings.chunk_overlap
        ):
            pending_chunks.append((chunk_text, page_number, chunk_index))
            chunk_index += 1

    if not pending_chunks:
        raise DocumentProcessingError("No readable text chunks were produced.")
    if len(pending_chunks) > max_chunks:
        raise DocumentProcessingError(
            "This document exceeds the remaining in-memory indexing limit. "
            "Upload a smaller document or remove indexed documents."
        )

    vectors = embed_texts(
        client,
        settings,
        [text for text, _, _ in pending_chunks],
        progress_callback=progress_callback,
    )
    if len(vectors) != len(pending_chunks):
        raise EmbeddingError("Gemini returned an incomplete embedding response.")

    chunks = tuple(
        DocumentChunk(
            text=text,
            source_filename=filename,
            page_number=page_number,
            chunk_index=index,
            embedding=vector,
        )
        for (text, page_number, index), vector in zip(pending_chunks, vectors)
    )
    return IndexedDocument(
        document_id=document_id_for_upload(filename, file_bytes),
        filename=filename,
        chunks=chunks,
    )


def add_document_to_session(
    state: MutableMapping[str, Any], document: IndexedDocument
) -> None:
    """Add a fully indexed document to only the supplied session state."""
    state[DOCUMENTS_KEY][document.document_id] = document
    state[DOCUMENT_STATUSES_KEY][document.document_id] = {
        "filename": document.filename,
        "status": "Indexed",
        "message": "",
    }


def remove_document_from_session(
    state: MutableMapping[str, Any], document_id: str
) -> None:
    """Remove an indexed document and its status from the supplied session."""
    state[DOCUMENTS_KEY].pop(document_id, None)
    state[DOCUMENT_STATUSES_KEY].pop(document_id, None)


def clear_session_documents(state: MutableMapping[str, Any]) -> None:
    """Remove all indexed documents and per-document statuses from the session."""
    state[DOCUMENTS_KEY].clear()
    state[DOCUMENT_STATUSES_KEY].clear()


def set_document_status(
    state: MutableMapping[str, Any],
    document_id: str,
    filename: str,
    status: str,
    message: str = "",
) -> None:
    """Store a user-visible processing status without storing file contents."""
    state[DOCUMENT_STATUSES_KEY][document_id] = {
        "filename": filename,
        "status": status,
        "message": message,
    }


def _cosine_similarity(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def retrieve_relevant_chunks(
    client: genai.Client,
    settings: Settings,
    question: str,
    documents: Mapping[str, IndexedDocument],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Embed a query and rank this session's document chunks by cosine similarity."""
    count = settings.retriever_top_k if top_k is None else top_k
    if count < 1:
        raise ValueError("top_k must be at least 1")
    if not question.strip() or not documents:
        return []

    query_vector = embed_texts(
        client, settings, [question], task_type="RETRIEVAL_QUERY"
    )[0]
    ranked = [
        RetrievedChunk(chunk, _cosine_similarity(query_vector, chunk.embedding))
        for document in documents.values()
        for chunk in document.chunks
    ]
    ranked.sort(key=lambda result: result.similarity, reverse=True)
    return ranked[:count]


def format_source_citations(results: Sequence[RetrievedChunk]) -> list[str]:
    """Return unique filename/page citations in retrieval order."""
    citations: list[str] = []
    for result in results:
        chunk = result.chunk
        citation = chunk.source_filename
        if chunk.page_number is not None:
            citation += f", page {chunk.page_number}"
        if citation not in citations:
            citations.append(citation)
    return citations
