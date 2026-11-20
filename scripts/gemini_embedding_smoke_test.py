"""Make one live, configured Gemini embedding request for manual diagnosis."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Verify one configured retrieval-document embedding without showing secrets."""
    from chatbot.chat_engine import (
        GeminiConfigurationError,
        GeminiRequestError,
        create_gemini_client,
    )
    from chatbot.config import load_settings
    from chatbot.rag_integration import EmbeddingError, embed_texts

    settings = load_settings()
    print(f"Embedding model: {settings.gemini_embedding_model}")
    print(f"Configured dimensions: {settings.gemini_embedding_dimensions}")

    try:
        client = create_gemini_client(settings)
        with client:
            vectors = embed_texts(
                client,
                settings,
                ["DocuChat embedding connectivity check."],
                task_type="RETRIEVAL_DOCUMENT",
            )
    except GeminiConfigurationError:
        print("Embedding smoke test failed: GEMINI_API_KEY is not configured.")
        return 1
    except GeminiRequestError as exc:
        print(
            "Embedding smoke test failed during client setup "
            f"(HTTP {exc.status_code or 'unavailable'})."
        )
        return 1
    except EmbeddingError as exc:
        print(f"Embedding smoke test failed: {exc}")
        print(f"Failure category: {exc.category}")
        if exc.status_code is not None:
            print(f"HTTP status: {exc.status_code}")
        return 1

    if not vectors or not vectors[0]:
        print("Embedding smoke test failed: no vector was returned.")
        return 1
    actual_dimensions = len(vectors[0])
    if actual_dimensions != settings.gemini_embedding_dimensions:
        print(
            "Embedding smoke test failed: returned dimensions do not match "
            f"the configured value ({actual_dimensions})."
        )
        return 1

    print(f"Returned dimensions: {actual_dimensions}")
    print("Embedding smoke test: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
