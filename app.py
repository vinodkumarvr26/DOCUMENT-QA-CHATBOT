"""DocuChat AI Streamlit chat application."""

from html import escape

import streamlit as st

from chatbot import chat_engine, config, stream_handler
from chatbot.conversation_memory import (
    SESSION_MESSAGES_KEY,
    append_assistant_message,
    append_user_message,
    clear_conversation,
    enforce_history_limit,
    get_completed_turns,
    initialize_conversation_state,
    start_new_conversation,
)
from chatbot import rag_integration


UPLOAD_WIDGET_VERSION_KEY = "document_upload_widget_version"
ACTIVE_VIEW_KEY = "active_view"
PENDING_PROMPT_KEY = "pending_chat_prompt"
MAX_UPLOAD_FILES = 5
MAX_UPLOAD_FILE_SIZE_MB = 10
SUGGESTIONS = (
    ("Study Notes", "Help me study. Ask which subject or notes I want to work on."),
    (
        "Summarize Documents",
        "Summarize the key points from my uploaded documents.",
    ),
    (
        "Ask Questions",
        "I have a question. Ask what I would like to know and help me explore it.",
    ),
    (
        "Brainstorm Ideas",
        "Help me brainstorm ideas for a project and ask about my goals if needed.",
    ),
    (
        "Explain a Topic",
        "Help me understand a topic clearly. Ask which topic I have in mind.",
    ),
    (
        "Research Assistance",
        "Help me explore a topic conversationally. Ask what I want to research.",
    ),
)
FEATURES = (
    (
        "Document Q&A",
        "Ask questions grounded in your uploaded files.",
        "documents",
    ),
    (
        "Study & Learn",
        "Break down a topic and learn it step by step.",
        "Help me study a topic. Ask what I am learning and where I need help.",
    ),
    (
        "Summarization",
        "Get a concise overview of your indexed documents.",
        "summarize",
    ),
    (
        "Writing & Planning",
        "Organize your ideas into a practical plan.",
        "Help me organize my ideas into a clear, practical plan.",
    ),
    (
        "Research Assistant",
        "Explore a subject through a focused conversation.",
        "Help me explore a topic conversationally. Ask what I want to research.",
    ),
)


def _apply_styles() -> None:
    """Apply the dark navy and electric-blue DocuChat design system."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --dc-bg: #070d19;
            --dc-surface: #0c1627;
            --dc-sidebar: #091323;
            --dc-card: #15243a;
            --dc-card-raised: #1b2d46;
            --dc-blue: #75a9ff;
            --dc-electric: #4b8dff;
            --dc-highlight: #a2c4ff;
            --dc-text: #f5f7fc;
            --dc-muted: #a6b4c8;
            --dc-border: #293d59;
        }
        html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
            font-family: Inter, Geist, Manrope, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: var(--dc-bg);
            color: var(--dc-text);
        }
        [data-testid="stHeader"] {
            background: rgba(7, 13, 25, 0.9);
        }
        [data-testid="stIconMaterial"] {
            display: none !important;
        }
        [data-testid="stExpandSidebarButton"],
        button[aria-label="Collapse sidebar"] {
            color: var(--dc-blue);
            font-size: 0;
        }
        [data-testid="stExpandSidebarButton"]::after {
            content: "";
            display: block;
            width: 16px;
            height: 12px;
            background:
                linear-gradient(currentColor, currentColor) top / 100% 2px no-repeat,
                linear-gradient(currentColor, currentColor) center / 100% 2px no-repeat,
                linear-gradient(currentColor, currentColor) bottom / 100% 2px no-repeat;
        }
        button[aria-label="Collapse sidebar"]::after {
            content: "";
            display: block;
            width: 9px;
            height: 9px;
            border-left: 2px solid currentColor;
            border-bottom: 2px solid currentColor;
            transform: rotate(45deg);
        }
        [data-testid="stAppViewBlockContainer"] {
            max-width: 1540px;
            padding: 1.5rem 2rem 7rem;
        }
        [data-testid="stSidebar"] {
            background: var(--dc-sidebar);
            border-right: 1px solid rgba(117, 169, 255, 0.12);
        }
        [data-testid="stSidebarContent"] {
            padding: 1.1rem 0.9rem 1.5rem;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] small {
            color: var(--dc-muted);
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
            min-height: 2.7rem;
            color: var(--dc-muted);
            border: 1px solid transparent;
            background: transparent;
            text-align: left;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            min-height: 2.7rem;
            color: #07101f;
            border: 1px solid var(--dc-blue);
            background: linear-gradient(110deg, var(--dc-blue), #93baff);
            text-align: left;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            color: var(--dc-text);
            border-color: var(--dc-border);
            background: rgba(117, 169, 255, 0.08);
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--dc-border);
        }
        [data-testid="stSidebar"] [data-testid="stAlert"] {
            background: rgba(75, 141, 255, 0.12);
            border-color: rgba(117, 169, 255, 0.3);
        }
        [data-testid="stSidebar"] [data-testid="stAlert"] * {
            color: var(--dc-text);
        }
        h1, h2, h3, h4, p, label, li {
            color: var(--dc-text);
        }
        h1 {
            letter-spacing: -0.045em;
        }
        [data-testid="stCaptionContainer"], .stCaption {
            color: var(--dc-muted);
            font-size: 0.875rem;
        }
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-tertiary"] {
            border-radius: 12px;
            transition: background-color 140ms ease, border-color 140ms ease,
                transform 140ms ease, box-shadow 140ms ease;
        }
        [data-testid="stBaseButton-secondary"] {
            color: var(--dc-text);
            border-color: var(--dc-border);
            background: rgba(21, 36, 58, 0.82);
        }
        [data-testid="stBaseButton-secondary"]:hover {
            color: var(--dc-text);
            border-color: rgba(117, 169, 255, 0.65);
            background: var(--dc-card-raised);
            transform: translateY(-1px);
        }
        [data-testid="stBaseButton-primary"] {
            color: #07101f;
            border-color: var(--dc-blue);
            background: linear-gradient(110deg, #75a9ff, #a2c4ff);
            box-shadow: 0 8px 24px rgba(75, 141, 255, 0.2);
        }
        [data-testid="stBaseButton-primary"]:hover {
            color: #07101f;
            border-color: #c3d8ff;
            background: linear-gradient(110deg, #8cb8ff, #bfd5ff);
            box-shadow: 0 10px 28px rgba(75, 141, 255, 0.28);
        }
        [data-testid="stBaseButton-secondary"]:focus-visible,
        [data-testid="stBaseButton-primary"]:focus-visible,
        [data-testid="stChatInput"] textarea:focus {
            outline: 2px solid var(--dc-highlight);
            outline-offset: 2px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--dc-border);
            border-radius: 18px;
            background: linear-gradient(
                145deg, rgba(21, 36, 58, 0.94), rgba(12, 22, 39, 0.98)
            );
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.16);
        }
        [data-testid="stChatMessage"] {
            border: 1px solid var(--dc-border);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: var(--dc-card);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            width: min(88%, 56rem);
            margin-left: auto;
            border-color: rgba(117, 169, 255, 0.36);
            background: linear-gradient(130deg, #477fda, #75a9ff);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
            [data-testid="stMarkdownContainer"] * {
            color: #081426;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            width: min(96%, 62rem);
            margin-right: auto;
            background: linear-gradient(145deg, #15243a, #101c30);
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.72;
        }
        [data-testid="stChatInput"] {
            width: min(69%, 880px);
            border: 1px solid var(--dc-border);
            border-radius: 18px;
            background: rgba(12, 22, 39, 0.97);
            box-shadow: 0 10px 36px rgba(0, 0, 0, 0.32);
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--dc-electric);
            box-shadow: 0 0 0 1px rgba(75, 141, 255, 0.35),
                0 12px 40px rgba(0, 0, 0, 0.38);
        }
        [data-testid="stChatInput"] textarea {
            color: var(--dc-text);
            caret-color: var(--dc-highlight);
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #7f91aa;
        }
        [data-testid="stChatInput"] button {
            color: #07101f;
            background: var(--dc-blue);
            border-radius: 12px;
        }
        [data-testid="stFileUploader"] section {
            border: 1px dashed #47648a;
            border-radius: 16px;
            background: rgba(7, 13, 25, 0.36);
        }
        [data-testid="stFileUploader"] button {
            border-color: var(--dc-border);
            color: var(--dc-text);
            background: var(--dc-card);
        }
        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, var(--dc-electric), var(--dc-highlight));
        }
        [data-testid="stAlert"] {
            border-radius: 14px;
            border-color: var(--dc-border);
            background: #122139;
        }
        [data-testid="stAlert"] * {
            color: var(--dc-text);
        }
        .dc-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0.4rem 0 0.25rem;
        }
        .dc-brand-mark {
            display: grid;
            width: 2.25rem;
            height: 2.25rem;
            place-items: center;
            border: 1px solid rgba(162, 196, 255, 0.68);
            border-radius: 11px;
            color: #07101f;
            background: linear-gradient(145deg, #a2c4ff, #4b8dff);
            font-size: 1.05rem;
            font-weight: 800;
            box-shadow: 0 5px 18px rgba(75, 141, 255, 0.25);
        }
        .dc-brand-name {
            color: var(--dc-text);
            font-size: 1.03rem;
            font-weight: 750;
            letter-spacing: -0.02em;
        }
        .dc-brand-caption {
            color: #8395ad;
            font-size: 0.875rem;
            margin-top: 0.08rem;
        }
        .dc-kicker {
            color: var(--dc-blue);
            font-size: 0.875rem;
            font-weight: 750;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }
        .dc-muted {
            color: var(--dc-muted);
        }
        .dc-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.2rem;
        }
        .dc-status-pill, .dc-model-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.38rem 0.7rem;
            border: 1px solid var(--dc-border);
            border-radius: 999px;
            color: var(--dc-muted);
            background: rgba(12, 22, 39, 0.85);
            font-size: 0.875rem;
        }
        .dc-status-dot {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: #71d5ae;
            box-shadow: 0 0 12px rgba(113, 213, 174, 0.65);
        }
        .dc-status-dot.offline {
            background: #e9b46b;
            box-shadow: 0 0 12px rgba(233, 180, 107, 0.45);
        }
        .dc-hero {
            position: relative;
            overflow: hidden;
            min-height: 18rem;
            padding: clamp(1.5rem, 3.2vw, 3rem);
            border: 1px solid rgba(117, 169, 255, 0.3);
            border-radius: 26px;
            background:
                radial-gradient(ellipse at 82% 18%, rgba(75, 141, 255, 0.34), transparent 38%),
                radial-gradient(ellipse at 18% 110%, rgba(41, 96, 177, 0.3), transparent 45%),
                linear-gradient(125deg, #0c1627 10%, #10213b 56%, #0a1424);
            box-shadow: inset 0 1px rgba(255,255,255,0.04),
                0 24px 60px rgba(0, 0, 0, 0.28);
        }
        .dc-hero::before {
            position: absolute;
            inset: 0 0 0 38%;
            background-image: radial-gradient(
                circle, rgba(162, 196, 255, 0.6) 0.75px, transparent 1px
            );
            background-size: 9px 9px;
            content: "";
            opacity: 0.19;
            mask-image: linear-gradient(110deg, transparent, #000 55%, transparent);
            pointer-events: none;
        }
        .dc-hero-content {
            position: relative;
            z-index: 1;
            max-width: 46rem;
        }
        .dc-hero h1 {
            max-width: 13ch;
            margin: 0.9rem 0 0.8rem;
            color: var(--dc-text);
            font-size: clamp(2.2rem, 4.8vw, 4.35rem);
            font-weight: 760;
            line-height: 1.03;
            letter-spacing: -0.06em;
        }
        .dc-hero h1 span {
            color: var(--dc-highlight);
        }
        .dc-hero p {
            max-width: 38rem;
            margin: 0;
            color: #b6c4d8;
            font-size: 1.02rem;
            line-height: 1.65;
        }
        .dc-section-title {
            margin: 1.7rem 0 0.75rem;
            color: var(--dc-text);
            font-size: 1.1rem;
            font-weight: 680;
            letter-spacing: -0.02em;
        }
        .dc-thinking {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            color: var(--dc-muted);
            font-size: 0.85rem;
        }
        .dc-thinking-dots {
            display: inline-flex;
            gap: 0.2rem;
        }
        .dc-thinking-dots i {
            width: 0.28rem;
            height: 0.28rem;
            border-radius: 50%;
            background: var(--dc-blue);
            animation: dc-pulse 1.05s ease-in-out infinite;
        }
        .dc-thinking-dots i:nth-child(2) { animation-delay: 0.15s; }
        .dc-thinking-dots i:nth-child(3) { animation-delay: 0.3s; }
        @keyframes dc-pulse {
            0%, 65%, 100% { opacity: 0.35; transform: translateY(0); }
            32% { opacity: 1; transform: translateY(-3px); }
        }
        .dc-citation-label {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.55rem;
            border: 1px solid rgba(117, 169, 255, 0.3);
            border-radius: 999px;
            color: var(--dc-highlight);
            background: rgba(75, 141, 255, 0.1);
            font-size: 0.875rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .dc-document-title {
            overflow-wrap: anywhere;
            color: var(--dc-text);
            font-size: 0.875rem;
            font-weight: 620;
        }
        .dc-document-meta {
            color: var(--dc-muted);
            font-size: 0.875rem;
        }
        .dc-source-card {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            max-width: 100%;
            margin: 0.3rem 0.35rem 0.15rem 0;
            padding: 0.42rem 0.65rem;
            overflow-wrap: anywhere;
            border: 1px solid rgba(117, 169, 255, 0.28);
            border-radius: 12px;
            color: var(--dc-highlight);
            background: rgba(75, 141, 255, 0.09);
            font-size: 0.875rem;
        }
        .dc-status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.55rem;
            border: 1px solid var(--dc-border);
            border-radius: 999px;
            color: var(--dc-muted);
            background: rgba(12, 22, 39, 0.8);
            font-size: 0.875rem;
        }
        .dc-status-badge.indexed {
            border-color: rgba(113, 213, 174, 0.35);
            color: #a4e5cb;
        }
        .dc-status-badge.failed {
            border-color: rgba(242, 132, 132, 0.35);
            color: #ffb1b1;
        }
        .dc-status-badge.indexing {
            border-color: rgba(117, 169, 255, 0.4);
            color: var(--dc-highlight);
        }
        @media (max-width: 900px) {
            [data-testid="stAppViewBlockContainer"] {
                padding: 1rem 1.15rem 7rem;
            }
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.7rem;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100%;
                min-width: 100%;
            }
            .dc-hero {
                min-height: auto;
                border-radius: 22px;
            }
            .dc-hero h1 {
                font-size: clamp(2.15rem, 9vw, 3.25rem);
            }
            [data-testid="stChatInput"] {
                width: 100%;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }
        }
        @media (max-width: 480px) {
            [data-testid="stBottom"]::before {
                position: absolute;
                top: -2.5rem;
                right: 0;
                left: 0;
                height: 2.5rem;
                background: linear-gradient(
                    180deg, transparent, var(--dc-bg) 40%
                );
                content: "";
                pointer-events: none;
            }
            [data-testid="stAppViewBlockContainer"] {
                padding-right: 0.85rem;
                padding-left: 0.85rem;
            }
            [data-testid="stChatMessage"] {
                padding: 0.7rem 0.78rem;
                border-radius: 15px;
            }
            .dc-topline {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.6rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _friendly_request_error(error: chat_engine.GeminiRequestError) -> str:
    """Translate provider status into safe, actionable UI text."""
    if error.category == "unexpected":
        return "An unexpected error occurred while contacting Gemini. Please try again."
    return _friendly_status_error(error.status_code)


def _friendly_status_error(status_code: int | None) -> str:
    if status_code in (401, 403):
        return "Gemini rejected the configured credentials. Check your local API key."
    if status_code == 429:
        return "Gemini's quota or rate limit was reached. Please try again later."
    if status_code is not None and status_code >= 500:
        return "Gemini is temporarily unavailable. Please try again shortly."
    if status_code is None:
        return "Could not connect to Gemini. Check your network and try again."
    return "Gemini could not complete this request. Please try again."


def _friendly_embedding_error(error: rag_integration.EmbeddingError) -> str:
    messages = {
        "authentication": (
            "Gemini rejected the credentials. Check the API key configured locally."
        ),
        "unsupported_model": (
            "The configured Gemini embedding model is unavailable. "
            "Check GEMINI_EMBEDDING_MODEL."
        ),
        "invalid_request": (
            "Gemini rejected the embedding settings. Check the model, task type, "
            "and output dimensions."
        ),
        "quota": (
            "Gemini's embedding quota or rate limit was reached. Please try again later."
        ),
        "service": "Gemini's embedding service is temporarily unavailable.",
        "network": "Could not connect to Gemini. Check your network and try again.",
        "timeout": "The Gemini embedding request timed out. Please try again.",
        "invalid_response": (
            "Gemini returned an invalid embedding. Check the configured model and "
            "output dimensions."
        ),
    }
    return messages.get(
        error.category,
        "An unexpected error occurred while preparing document embeddings.",
    )


def _record_error(
    turn_id: str,
    message: str,
    history_limit: int,
    citations: tuple[str, ...] = (),
) -> None:
    append_assistant_message(
        st.session_state,
        turn_id,
        message,
        include_in_context=False,
        is_error=True,
        citations=citations,
    )
    enforce_history_limit(st.session_state, history_limit)
    st.error(message)


def _activate_chat_view() -> None:
    """Route submitted composer messages to the conversation screen."""
    st.session_state[ACTIVE_VIEW_KEY] = "chat"


def _handle_prompt(prompt: str, settings: config.Settings) -> None:
    """Store a prompt, stream its response, and save only completed turns."""
    turn_id = append_user_message(st.session_state, prompt)
    if turn_id is None:
        return

    with st.chat_message("user"):
        st.markdown(prompt.strip())

    with st.chat_message("assistant"):
        progress = st.empty()
        progress.markdown(
            '<div class="dc-thinking">DocuChat is thinking'
            '<span class="dc-thinking-dots"><i></i><i></i><i></i></span></div>',
            unsafe_allow_html=True,
        )
        citations: list[str] = []
        try:
            with chat_engine.create_gemini_client(settings) as client:
                indexed_documents = st.session_state[rag_integration.DOCUMENTS_KEY]
                if indexed_documents:
                    retrieved = rag_integration.retrieve_relevant_chunks(
                        client,
                        settings,
                        prompt.strip(),
                        indexed_documents,
                    )
                    citations = rag_integration.format_source_citations(retrieved)
                    response_stream = stream_handler.stream_document_response(
                        client,
                        settings,
                        prompt.strip(),
                        retrieved,
                    )
                else:
                    turns = get_completed_turns(st.session_state)
                    response_stream = stream_handler.stream_chat_response(
                        client, settings, turns, prompt.strip()
                    )
                response = st.write_stream(response_stream)
        except chat_engine.GeminiConfigurationError:
            progress.empty()
            _record_error(
                turn_id,
                "Gemini is not configured. Add GEMINI_API_KEY to your local environment.",
                settings.chat_history_limit,
            )
            return
        except chat_engine.GeminiRequestError as exc:
            progress.empty()
            _record_error(
                turn_id,
                _friendly_request_error(exc),
                settings.chat_history_limit,
            )
            return
        except rag_integration.EmbeddingError as exc:
            progress.empty()
            _record_error(
                turn_id,
                _friendly_embedding_error(exc),
                settings.chat_history_limit,
            )
            return

        progress.empty()
        if not isinstance(response, str) or not response.strip():
            _record_error(
                turn_id,
                "Gemini returned an empty response. Please try again.",
                settings.chat_history_limit,
                tuple(citations),
            )
            return

    append_assistant_message(
        st.session_state,
        turn_id,
        response,
        citations=tuple(citations),
    )
    enforce_history_limit(st.session_state, settings.chat_history_limit)


def main() -> None:
    """Render the premium assistant workspace and process chat submissions."""
    st.set_page_config(page_title="DocuChat AI", page_icon="D", layout="wide")
    settings = config.load_settings()
    initialize_conversation_state(st.session_state)
    rag_integration.initialize_document_state(st.session_state)
    st.session_state.setdefault(ACTIVE_VIEW_KEY, "home")
    _apply_styles()

    api_key_configured = bool(
        settings.gemini_api_key and settings.gemini_api_key.strip()
    )
    with st.sidebar:
        st.markdown(
            '<div class="dc-brand"><div class="dc-brand-mark">D</div>'
            '<div><div class="dc-brand-name">DocuChat AI</div>'
            '<div class="dc-brand-caption">Understand more, together</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("##### WORKSPACE")
        for label, view in (
            ("Home", "home"),
            ("Chat", "chat"),
            ("Documents", "documents"),
        ):
            if st.button(
                label,
                key=f"nav_{view}",
                type=(
                    "primary"
                    if st.session_state[ACTIVE_VIEW_KEY] == view
                    else "secondary"
                ),
                use_container_width=True,
            ):
                st.session_state[ACTIVE_VIEW_KEY] = view
        st.divider()
        st.markdown("##### CONVERSATION")
        if st.button(
            "New chat",
            key="new_chat",
            type="primary",
            use_container_width=True,
        ):
            start_new_conversation(st.session_state)
            st.session_state[ACTIVE_VIEW_KEY] = "chat"
            st.rerun()
        if st.button(
            "Clear conversation",
            key="clear_conversation",
            disabled=not st.session_state[SESSION_MESSAGES_KEY],
            use_container_width=True,
        ):
            clear_conversation(st.session_state)
            st.rerun()

        st.divider()
        status_class = "" if api_key_configured else "offline"
        status_text = "Gemini connected" if api_key_configured else "API key required"
        st.markdown(
            f'<div class="dc-status-pill"><span class="dc-status-dot '
            f'{status_class}"></span>{status_text}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model · {settings.gemini_chat_model}")
        st.caption(
            f"{len(get_completed_turns(st.session_state))} completed turns · "
            f"up to {settings.chat_history_limit} retained"
        )
        st.caption("Session data is isolated and stored in memory.")

    content_column, context_column = st.columns([2.15, 0.95], gap="large")
    suggested_prompt = None
    with content_column:
        if not api_key_configured:
            st.info("Set `GEMINI_API_KEY` in the environment to enable Gemini.")

        view = st.session_state[ACTIVE_VIEW_KEY]
        if view == "home":
            suggested_prompt = _render_home(api_key_configured)
        elif view == "documents":
            _render_document_workspace()
        else:
            suggested_prompt = _render_conversation(
                settings, api_key_configured
            )

    with context_column:
        _render_context_panel(settings, api_key_configured)

    submitted_prompt = st.chat_input(
        "Ask anything or explore your documents...",
        key="chat_input",
        disabled=not api_key_configured,
        on_submit=_activate_chat_view,
    )
    if suggested_prompt is not None:
        st.session_state[PENDING_PROMPT_KEY] = suggested_prompt
        st.session_state[ACTIVE_VIEW_KEY] = "chat"
        st.rerun()

    pending_prompt = st.session_state.pop(PENDING_PROMPT_KEY, None)
    prompt = pending_prompt or submitted_prompt
    if prompt is not None and api_key_configured:
        st.session_state[ACTIVE_VIEW_KEY] = "chat"
        _handle_prompt(prompt, settings)
        st.rerun()


def _render_home(api_key_configured: bool) -> str | None:
    """Render the welcome dashboard and return any selected chat prompt."""
    st.markdown(
        """
        <section class="dc-hero">
          <div class="dc-hero-content">
            <div class="dc-kicker">YOUR AI WORKSPACE</div>
            <h1>One assistant.<br><span>Many ways to understand.</span></h1>
            <p>Explore ideas, understand documents, and get meaningful answers
               with DocuChat AI.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "Start Chatting",
        key="start_chatting",
        type="primary",
        disabled=not api_key_configured,
    ):
        st.session_state[ACTIVE_VIEW_KEY] = "chat"
        st.rerun()
    st.markdown('<div class="dc-section-title">Start with an idea</div>',
                unsafe_allow_html=True)
    selected_prompt = None
    for row_index in range(0, len(SUGGESTIONS), 3):
        columns = st.columns(3, gap="small")
        for column, (label, prompt) in zip(
            columns, SUGGESTIONS[row_index : row_index + 3]
        ):
            with column:
                if st.button(
                    label,
                    key=f"home_suggestion_{label}",
                    disabled=not api_key_configured,
                    use_container_width=True,
                ):
                    if label == "Summarize Documents" and not st.session_state[
                        rag_integration.DOCUMENTS_KEY
                    ]:
                        st.session_state[ACTIVE_VIEW_KEY] = "documents"
                        st.rerun()
                    else:
                        selected_prompt = prompt

    st.markdown('<div class="dc-section-title">A little help with a lot</div>',
                unsafe_allow_html=True)
    for row_index in range(0, len(FEATURES), 2):
        columns = st.columns(2, gap="medium")
        for column, feature in zip(columns, FEATURES[row_index : row_index + 2]):
            title, description, action = feature
            with column:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="dc-kicker">{description}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        title,
                        key=f"feature_{action}",
                        disabled=not api_key_configured and action != "documents",
                        use_container_width=True,
                    ):
                        if action == "documents" or (
                            action == "summarize"
                            and not st.session_state[rag_integration.DOCUMENTS_KEY]
                        ):
                            st.session_state[ACTIVE_VIEW_KEY] = "documents"
                            st.rerun()
                        elif action == "summarize":
                            selected_prompt = (
                                "Summarize the key points from my uploaded documents."
                            )
                        else:
                            selected_prompt = action
    if selected_prompt is not None:
        st.session_state[ACTIVE_VIEW_KEY] = "chat"
    return selected_prompt


def _render_conversation(
    settings: config.Settings, api_key_configured: bool
) -> str | None:
    """Render conversation history, citations, and its empty-state suggestions."""
    messages = st.session_state[SESSION_MESSAGES_KEY]
    st.markdown(
        '<div class="dc-topline"><span class="dc-kicker">CONVERSATION</span>'
        f'<span class="dc-model-pill">{escape(settings.gemini_chat_model)}</span></div>',
        unsafe_allow_html=True,
    )
    st.title("Your conversation")
    if not messages:
        st.markdown(
            '<p class="dc-muted">Pick up where you left off, or start with a prompt.</p>',
            unsafe_allow_html=True,
        )

    for message in messages:
        with st.chat_message(message["role"]):
            if message["is_error"]:
                st.error(message["content"])
            else:
                st.markdown(message["content"])
                if message.get("citations"):
                    st.markdown(
                        '<span class="dc-citation-label">Document sources</span>',
                        unsafe_allow_html=True,
                    )
                    for citation in message["citations"]:
                        st.markdown(
                            '<span class="dc-source-card">Source · '
                            f'{escape(citation)}</span>',
                            unsafe_allow_html=True,
                        )

    if messages:
        return None

    st.markdown('<div class="dc-section-title">Try a conversation starter</div>',
                unsafe_allow_html=True)
    for row_index in range(0, 4, 2):
        columns = st.columns(2, gap="small")
        for column, (label, prompt) in zip(
            columns, SUGGESTIONS[row_index : row_index + 2]
        ):
            with column:
                if st.button(
                    label,
                    key=f"chat_suggestion_{label}",
                    disabled=not api_key_configured,
                    use_container_width=True,
                ):
                    if label == "Summarize Documents" and not st.session_state[
                        rag_integration.DOCUMENTS_KEY
                    ]:
                        st.session_state[ACTIVE_VIEW_KEY] = "documents"
                        st.rerun()
                    return prompt
    return None


def _render_document_workspace() -> None:
    """Explain the session-scoped document workflow in the main workspace."""
    documents = st.session_state[rag_integration.DOCUMENTS_KEY]
    statuses = st.session_state[rag_integration.DOCUMENT_STATUSES_KEY]
    chunk_count = sum(len(document.chunks) for document in documents.values())
    st.markdown(
        '<div class="dc-topline"><span class="dc-kicker">YOUR KNOWLEDGE SPACE</span>'
        '<span class="dc-model-pill">PDF · TXT · session-only</span></div>',
        unsafe_allow_html=True,
    )
    st.title("Documents")
    st.markdown(
        '<p class="dc-muted">Add files to ask questions grounded in their contents. '
        'Indexes remain in this browser session only.</p>',
        unsafe_allow_html=True,
    )
    metrics = st.columns(3)
    metrics[0].metric("Indexed documents", len(documents))
    metrics[1].metric("Indexed chunks", chunk_count)
    metrics[2].metric("Session limit", rag_integration.MAX_INDEXED_CHUNKS_PER_SESSION)
    with st.container(border=True):
        st.markdown("#### Your document workflow")
        st.write(
            "Upload PDF or UTF-8 TXT files in the right-hand panel, then select "
            "**Index documents**. DocuChat extracts text, creates Gemini embeddings, "
            "and retrieves relevant passages when you ask a question."
        )
        st.caption(
            "Scanned PDFs are not supported. Up to five files can be indexed at once; "
            "each file is limited to 10 MB and 250 text chunks."
        )
    if statuses:
        st.markdown('<div class="dc-section-title">Recent file activity</div>',
                    unsafe_allow_html=True)
        for status in statuses.values():
            with st.container(border=True):
                status_class = {
                    "Indexed": "indexed",
                    "Failed": "failed",
                    "Indexing": "indexing",
                }.get(status["status"], "")
                st.markdown(
                    f'<div class="dc-document-title">'
                    f'{escape(status["filename"])}</div>'
                    f'<div class="dc-status-badge {status_class}">'
                    f'{escape(status["status"])}</div>',
                    unsafe_allow_html=True,
                )
                if status["message"]:
                    st.caption(status["message"])


def _render_context_panel(
    settings: config.Settings, api_key_configured: bool
) -> None:
    """Render the contextual document manager beside the main workspace."""
    documents = st.session_state[rag_integration.DOCUMENTS_KEY]
    chunks = sum(len(document.chunks) for document in documents.values())
    st.markdown(
        '<div class="dc-kicker">YOUR CONTEXT</div>'
        '<div class="dc-section-title">Your documents. Your answers.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="dc-muted">Upload files and let AI help you understand them.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="dc-status-pill">{len(documents)} indexed · {chunks} chunks</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="dc-document-meta">Gemini · '
        f'{escape(settings.gemini_chat_model)}</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Upload Documents",
        key="context_upload_documents",
        use_container_width=True,
    ):
        st.session_state[ACTIVE_VIEW_KEY] = "documents"
        st.rerun()
    _render_document_panel(settings, api_key_configured)


def _render_document_panel(
    settings: config.Settings, api_key_configured: bool
) -> None:
    """Render upload, indexing, status, removal, and clear controls."""
    documents = st.session_state[rag_integration.DOCUMENTS_KEY]
    statuses = st.session_state[rag_integration.DOCUMENT_STATUSES_KEY]
    st.session_state.setdefault(UPLOAD_WIDGET_VERSION_KEY, 0)
    with st.expander(f"Upload and index · {len(documents)} indexed", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            max_upload_size=MAX_UPLOAD_FILE_SIZE_MB,
            key=(
                f"document_upload_{st.session_state[UPLOAD_WIDGET_VERSION_KEY]}"
            ),
            help=(
                f"Up to {MAX_UPLOAD_FILES} files, {MAX_UPLOAD_FILE_SIZE_MB} MB each. "
                "Files are processed in memory and isolated to this browser session."
            ),
        )
        too_many_uploads = bool(
            uploaded_files and len(uploaded_files) > MAX_UPLOAD_FILES
        )
        if too_many_uploads:
            st.warning(f"Select no more than {MAX_UPLOAD_FILES} documents at a time.")
        if st.button(
            "Index documents",
            key="index_documents",
            disabled=(
                not uploaded_files or not api_key_configured or too_many_uploads
            ),
            use_container_width=True,
        ):
            try:
                client_context = chat_engine.create_gemini_client(settings)
            except chat_engine.GeminiConfigurationError:
                st.error("Set GEMINI_API_KEY locally before indexing documents.")
            except chat_engine.GeminiRequestError as exc:
                st.error(_friendly_request_error(exc))
            else:
                with client_context as client:
                    progress = st.progress(
                        0.0, text="Preparing document indexing..."
                    )
                    for index, uploaded_file in enumerate(uploaded_files):
                        if uploaded_file.size > MAX_UPLOAD_FILE_SIZE_MB * 1024 * 1024:
                            document_id = rag_integration.document_id_for_upload(
                                uploaded_file.name, uploaded_file.getvalue()
                            )
                            rag_integration.set_document_status(
                                st.session_state,
                                document_id,
                                uploaded_file.name,
                                "Failed",
                                f"Files must be no larger than "
                                f"{MAX_UPLOAD_FILE_SIZE_MB} MB.",
                            )
                            progress.progress(
                                (index + 1) / len(uploaded_files),
                                text=f"Rejected oversized file: {uploaded_file.name}",
                            )
                            continue
                        file_bytes = uploaded_file.getvalue()
                        document_id = rag_integration.document_id_for_upload(
                            uploaded_file.name, file_bytes
                        )
                        if document_id in documents:
                            progress.progress(
                                (index + 1) / len(uploaded_files),
                                text=f"Already indexed: {uploaded_file.name}",
                            )
                            continue
                        rag_integration.set_document_status(
                            st.session_state,
                            document_id,
                            uploaded_file.name,
                            "Indexing",
                        )
                        try:
                            indexed_chunks = sum(
                                len(document.chunks)
                                for document in documents.values()
                            )

                            def update_embedding_progress(
                                completed: int, total: int
                            ) -> None:
                                file_fraction = completed / total
                                overall_fraction = (
                                    index + file_fraction
                                ) / len(uploaded_files)
                                progress.progress(
                                    overall_fraction,
                                    text=(
                                        f"Indexing {uploaded_file.name}: "
                                        f"{completed}/{total} chunks"
                                    ),
                                )

                            indexed = rag_integration.index_document(
                                client,
                                settings,
                                uploaded_file.name,
                                file_bytes,
                                max_chunks=min(
                                    rag_integration.MAX_CHUNKS_PER_DOCUMENT,
                                    rag_integration.MAX_INDEXED_CHUNKS_PER_SESSION
                                    - indexed_chunks,
                                ),
                                progress_callback=update_embedding_progress,
                            )
                        except rag_integration.DocumentProcessingError as exc:
                            rag_integration.set_document_status(
                                st.session_state,
                                document_id,
                                uploaded_file.name,
                                "Failed",
                                str(exc),
                            )
                        except rag_integration.EmbeddingError as exc:
                            rag_integration.set_document_status(
                                st.session_state,
                                document_id,
                                uploaded_file.name,
                                "Failed",
                                _friendly_embedding_error(exc),
                            )
                        else:
                            rag_integration.add_document_to_session(
                                st.session_state, indexed
                            )
                        progress.progress(
                            (index + 1) / len(uploaded_files),
                            text=f"Processed {uploaded_file.name}",
                        )
                    progress.empty()
                    st.rerun()

        if statuses:
            for document_id, status in list(statuses.items()):
                row, remove = st.columns([3, 2])
                with row:
                    status_class = {
                        "Indexed": "indexed",
                        "Failed": "failed",
                        "Indexing": "indexing",
                    }.get(status["status"], "")
                    st.markdown(
                        f'<div class="dc-document-title">'
                        f'{escape(status["filename"])}</div>'
                        f'<div class="dc-status-badge {status_class}">'
                        f'{escape(status["status"])}</div>',
                        unsafe_allow_html=True,
                    )
                    if status["message"]:
                        st.caption(status["message"])
                with remove:
                    if st.button(
                        "Remove",
                        key=f"remove_document_{document_id}",
                        use_container_width=True,
                    ):
                        rag_integration.remove_document_from_session(
                            st.session_state, document_id
                        )
                        st.session_state[UPLOAD_WIDGET_VERSION_KEY] += 1
                        st.rerun()
        if statuses or uploaded_files:
            if st.button(
                "Clear all documents",
                key="clear_documents",
                use_container_width=True,
            ):
                rag_integration.clear_session_documents(st.session_state)
                st.session_state[UPLOAD_WIDGET_VERSION_KEY] += 1
                st.rerun()
        if uploaded_files:
            st.markdown("**Selected files**")
            for uploaded_file in uploaded_files:
                document_id = rag_integration.document_id_for_upload(
                    uploaded_file.name, uploaded_file.getvalue()
                )
                status = statuses.get(document_id)
                state = status["status"] if status else "Ready to index"
                st.text(f"{uploaded_file.name} · {state}")


if __name__ == "__main__":
    main()
