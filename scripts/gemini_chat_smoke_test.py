"""Manual live Gemini connection check; never called automatically by pytest."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Send one short prompt and report the connection outcome."""
    from google.genai import errors

    from chatbot.chat_engine import (
        GeminiConfigurationError,
        GeminiRequestError,
        create_gemini_client,
    )
    from chatbot.config import load_settings

    settings = load_settings()
    try:
        client = create_gemini_client(settings)
    except (GeminiConfigurationError, GeminiRequestError) as exc:
        print(str(exc))
        return 1

    try:
        response = client.models.generate_content(
            model=settings.gemini_chat_model,
            contents="Reply with the word OK.",
        )
    except errors.ClientError as exc:
        if exc.code in (401, 403):
            print(f"Gemini authentication failed (HTTP {exc.code}).")
        elif exc.code == 429:
            print("Gemini quota or rate limit reached (HTTP 429).")
        else:
            print(f"Gemini request was rejected (HTTP {exc.code}).")
        return 1
    except errors.ServerError as exc:
        print(f"Gemini service error (HTTP {exc.code}).")
        return 1
    except errors.APIError as exc:
        print(f"Gemini API error (HTTP {exc.code}).")
        return 1

    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text.strip():
        print("Gemini connection successful.")
        print(response_text)
        return 0

    print("Gemini returned no text.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
