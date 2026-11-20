"""Direct Google Gen AI client and streaming helpers."""

from collections.abc import Iterator, Sequence
import logging

from google import genai
import httpx2
from google.genai import errors, types

from chatbot.config import Settings, load_settings

logger = logging.getLogger(__name__)


class GeminiConfigurationError(RuntimeError):
    """Raised when required Gemini client configuration is missing."""


class GeminiRequestError(RuntimeError):
    """Raised when Gemini rejects or cannot complete a request."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        category: str = "network",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.category = category


def create_gemini_client(settings: Settings | None = None) -> genai.Client:
    """Create a Google Gen AI client without exposing its API key."""
    app_settings = settings or load_settings()
    api_key = app_settings.gemini_api_key
    if not api_key or not api_key.strip():
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not configured. Set it in the local environment."
        )

    try:
        return genai.Client(api_key=api_key)
    except errors.APIError as exc:
        logger.error(
            "Gemini client initialization API failure: exception=%s status=%d",
            type(exc).__name__,
            exc.code,
        )
        raise GeminiRequestError(
            f"Gemini client initialization failed (HTTP {exc.code}).",
            status_code=exc.code,
        ) from None
    except (httpx2.RequestError, TimeoutError, ConnectionError, OSError) as exc:
        logger.error(
            "Gemini client initialization connection failure: exception=%s",
            type(exc).__name__,
        )
        raise GeminiRequestError(
            "Gemini client initialization could not reach the service.",
            category="network",
        ) from None
    except Exception as exc:
        logger.error(
            "Unexpected Gemini client initialization failure: exception=%s",
            type(exc).__name__,
        )
        raise GeminiRequestError(
            "An unexpected error occurred while initializing Gemini.",
            category="unexpected",
        ) from None


def stream_gemini_response(
    client: genai.Client,
    model_name: str,
    conversation_history: Sequence[types.Content],
    system_prompt: str,
) -> Iterator[str]:
    """Yield non-empty text chunks from a streamed Gemini response."""
    try:
        chunks = client.models.generate_content_stream(
            model=model_name,
            contents=conversation_history,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        for chunk in chunks:
            text = chunk.text
            if isinstance(text, str) and text:
                yield text
    except errors.APIError as exc:
        logger.error(
            "Gemini streaming API failure: model=%s exception=%s status=%d",
            model_name,
            type(exc).__name__,
            exc.code,
        )
        raise GeminiRequestError(
            f"Gemini streaming request failed "
            f"({type(exc).__name__}, HTTP {exc.code}).",
            status_code=exc.code,
            category="api",
        ) from None
    except (httpx2.RequestError, TimeoutError, ConnectionError, OSError) as exc:
        logger.error(
            "Gemini streaming connection failure: model=%s exception=%s",
            model_name,
            type(exc).__name__,
        )
        raise GeminiRequestError(
            f"Gemini streaming request could not reach the service "
            f"({type(exc).__name__}).",
            category="network",
        ) from None
    except Exception as exc:
        logger.error(
            "Unexpected Gemini streaming failure: model=%s exception=%s",
            model_name,
            type(exc).__name__,
        )
        raise GeminiRequestError(
            "An unexpected error occurred while generating a Gemini response.",
            category="unexpected",
        ) from None
