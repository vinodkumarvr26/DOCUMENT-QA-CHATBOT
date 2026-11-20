"""Offline tests for configuration loading."""

from pathlib import Path

from chatbot.config import load_settings


def test_load_settings_uses_safe_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    for name in (
        "GEMINI_API_KEY",
        "GEMINI_CHAT_MODEL",
        "CHAT_HISTORY_LIMIT",
        "RETRIEVER_TOP_K",
    ):
        monkeypatch.delenv(name, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "GEMINI_CHAT_MODEL=test-chat-model",
                "CHAT_HISTORY_LIMIT=12",
                "RETRIEVER_TOP_K=3",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.gemini_chat_model == "test-chat-model"
    assert settings.chat_history_limit == 12
    assert settings.retriever_top_k == 3
    assert settings.gemini_api_key is None


def test_load_settings_reads_embedding_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    for name in (
        "GEMINI_EMBEDDING_DIMENSIONS",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
    ):
        monkeypatch.delenv(name, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_EMBEDDING_DIMENSIONS=768\n"
        "CHUNK_SIZE=1000\n"
        "CHUNK_OVERLAP=150\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.gemini_embedding_dimensions == 768
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
