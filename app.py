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


CUSTOM_STYLESHEET = """@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&display=swap');

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

*, *::before, *::after {
    font-family: 'Nunito', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    box-sizing: border-box;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--dc-bg) !important;
    color: var(--dc-text) !important;
}

[data-testid="stHeader"] {
    background: rgba(7, 13, 25, 0.9) !important;
}

[data-testid="stIconMaterial"] {
    display: none !important;
}

/* 1. Main page heading: 28-32px */
h1, .dc-hero h1, [data-testid="stAppViewContainer"] h1 {
    font-size: 32px !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em !important;
    color: var(--dc-text) !important;
    margin: 0.5rem 0 0.8rem !important;
}

/* 2. Section headings: 20-24px */
h2, h3, .dc-section-title, [data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3 {
    font-size: 22px !important;
    font-weight: 700 !important;
    line-height: 1.3 !important;
    letter-spacing: -0.01em !important;
    color: var(--dc-text) !important;
    margin: 1.2rem 0 0.6rem !important;
}

/* 3. Card titles: 16-18px */
h4, h5, .dc-document-title, [data-testid="stAppViewContainer"] h4 {
    font-size: 18px !important;
    font-weight: 700 !important;
    line-height: 1.35 !important;
    color: var(--dc-text) !important;
}

/* 4. Body text: 16px */
body, p, li, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
    font-size: 16px !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
    color: var(--dc-text) !important;
}

/* 5. Chat messages: 16px */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    font-size: 16px !important;
    font-weight: 500 !important;
    line-height: 1.65 !important;
}

/* 6. Sidebar navigation: 16px */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* 7. Buttons: 16px */
button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-tertiary"] {
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

/* 8. Inputs: 16px */
[data-testid="stChatInput"] textarea,
input, textarea, select {
    font-size: 16px !important;
    font-weight: 400 !important;
    line-height: 1.5 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    font-size: 16px !important;
    font-weight: 400 !important;
    color: #7f91aa !important;
}

/* 9. Secondary descriptions: 14px */
.dc-muted, p.dc-muted, .dc-hero p {
    font-size: 14px !important;
    font-weight: 400 !important;
    line-height: 1.55 !important;
    color: var(--dc-muted) !important;
}

/* 10. Metadata and captions: 14px */
[data-testid="stCaptionContainer"],
.stCaption,
.dc-brand-caption,
.dc-document-meta,
.dc-status-pill,
.dc-model-pill,
.dc-kicker,
.dc-status-badge,
.dc-citation-label,
.dc-source-card,
.dc-thinking {
    font-size: 14px !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
}

/* Minimum font size guarantee: nothing below 14px */
* {
    min-font-size: 14px;
}

/* Layout container */
[data-testid="stAppViewBlockContainer"] {
    max-width: 1540px !important;
    padding: 1.5rem 2rem 14rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: var(--dc-sidebar) !important;
    border-right: 1px solid rgba(117, 169, 255, 0.12) !important;
}
[data-testid="stSidebarContent"] {
    padding: 1.25rem 1rem 1.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: var(--dc-muted);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    min-height: 2.6rem;
    color: var(--dc-muted);
    border: 1px solid transparent;
    background: transparent;
    text-align: left;
    margin-bottom: 0.25rem;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    min-height: 2.6rem;
    color: #07101f;
    border: 1px solid var(--dc-blue);
    background: linear-gradient(110deg, var(--dc-blue), #93baff);
    text-align: left;
    margin-bottom: 0.25rem;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    color: var(--dc-text);
    border-color: var(--dc-border);
    background: rgba(117, 169, 255, 0.08);
}
[data-testid="stSidebar"] hr {
    margin: 0.85rem 0;
    border-color: rgba(117, 169, 255, 0.12);
}

/* Button hover & focus */
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
    box-shadow: 0 6px 20px rgba(75, 141, 255, 0.2);
}
[data-testid="stBaseButton-primary"]:hover {
    color: #07101f;
    border-color: #c3d8ff;
    background: linear-gradient(110deg, #8cb8ff, #bfd5ff);
    box-shadow: 0 8px 24px rgba(75, 141, 255, 0.28);
}

/* Chat messages */
[data-testid="stChatMessageAvatarCustom"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarImage"] {
    display: none !important;
}

[data-testid="stChatMessage"] {
    display: flex !important;
    border-radius: 18px !important;
    padding: 0.85rem 1.25rem !important;
    margin-bottom: 0.9rem !important;
    box-sizing: border-box !important;
    clear: both !important;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    width: 100% !important;
}

/* User messages: right-aligned, max-width ~78%, electric-blue, white text, rounded corners */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="user" i]),
[data-testid="stChatMessage"]:has([aria-label*="from user" i]) {
    width: fit-content !important;
    max-width: 78% !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    border: 1px solid rgba(117, 169, 255, 0.45) !important;
    border-radius: 18px 18px 4px 18px !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    box-shadow: 0 4px 18px rgba(29, 78, 216, 0.3) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="user" i]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label*="from user" i]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="user" i]) [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"]:has([aria-label*="from user" i]) [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="user" i]) p,
[data-testid="stChatMessage"]:has([aria-label*="from user" i]) p {
    color: #ffffff !important;
}

/* Assistant messages: left-aligned, max-width ~92%, dark navy elevated background, readable Markdown */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="assistant" i]),
[data-testid="stChatMessage"]:has([aria-label*="from assistant" i]) {
    width: 100% !important;
    max-width: 92% !important;
    margin-right: auto !important;
    margin-left: 0 !important;
    border: 1px solid rgba(117, 169, 255, 0.22) !important;
    border-radius: 18px 18px 18px 4px !important;
    background: linear-gradient(145deg, #132238 0%, #0d1829 100%) !important;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="assistant" i]) [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"]:has([aria-label*="from assistant" i]) [data-testid="stMarkdownContainer"] * {
    color: var(--dc-text) !important;
}

/* Composer */
[data-testid="stBottom"] {
    background: linear-gradient(180deg, rgba(7, 13, 25, 0) 0%, rgba(7, 13, 25, 0.94) 28%, #070d19 100%) !important;
    padding-top: 1.5rem !important;
    padding-bottom: 1.25rem !important;
}
[data-testid="stBottom"] > div {
    max-width: 1540px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
[data-testid="stChatInput"] {
    width: 100% !important;
    max-width: calc(68% - 0.75rem) !important;
    margin-right: auto !important;
    margin-left: 0 !important;
    border: 1px solid var(--dc-border) !important;
    border-radius: 16px !important;
    background: rgba(13, 24, 41, 0.98) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--dc-electric) !important;
    box-shadow: 0 0 0 1px rgba(75, 141, 255, 0.4), 0 12px 36px rgba(0, 0, 0, 0.48) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--dc-text);
    caret-color: var(--dc-highlight);
}
[data-testid="stChatInput"] button {
    color: #07101f;
    background: var(--dc-blue);
    border-radius: 12px;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    border: 1px dashed #47648a;
    border-radius: 14px;
    background: rgba(7, 13, 25, 0.45);
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

/* Brand & Topline components */
.dc-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.2rem 0 0.5rem;
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
    font-size: 18px;
    font-weight: 800;
    box-shadow: 0 4px 16px rgba(75, 141, 255, 0.25);
}
.dc-brand-name {
    color: var(--dc-text);
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.dc-kicker {
    color: var(--dc-blue);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    white-space: nowrap;
    flex-shrink: 0;
}
.dc-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem 0.75rem;
    min-height: 2rem;
    margin-bottom: 0.6rem;
    width: 100%;
}
.dc-topline h2 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: var(--dc-text) !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}
.dc-chat-spacer {
    height: 3.5rem;
    width: 100%;
}
.dc-status-pill, .dc-model-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.75rem;
    border: 1px solid var(--dc-border);
    border-radius: 999px;
    color: var(--dc-muted);
    background: rgba(12, 22, 39, 0.85);
    white-space: nowrap;
    flex-shrink: 0;
}
.dc-status-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: #71d5ae;
    box-shadow: 0 0 10px rgba(113, 213, 174, 0.65);
}
.dc-status-dot.offline {
    background: #e9b46b;
    box-shadow: 0 0 10px rgba(233, 180, 107, 0.45);
}
.dc-hero {
    position: relative;
    overflow: hidden;
    padding: 2rem;
    border: 1px solid rgba(117, 169, 255, 0.28);
    border-radius: 20px;
    background:
        radial-gradient(ellipse at 82% 18%, rgba(75, 141, 255, 0.32), transparent 38%),
        radial-gradient(ellipse at 18% 110%, rgba(41, 96, 177, 0.28), transparent 45%),
        linear-gradient(125deg, #0c1627 10%, #10213b 56%, #0a1424);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.26);
    margin-bottom: 1.25rem;
}
.dc-hero h1 span {
    color: var(--dc-highlight);
}
.dc-citation-label {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.6rem;
    border: 1px solid rgba(117, 169, 255, 0.3);
    border-radius: 999px;
    color: var(--dc-highlight);
    background: rgba(75, 141, 255, 0.1);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.dc-document-title {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
    color: var(--dc-text);
    font-size: 16px !important;
    font-weight: 700 !important;
    line-height: 1.4 !important;
    margin-bottom: 0.25rem;
}
.dc-source-card {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    max-width: 100%;
    margin: 0.3rem 0.35rem 0.15rem 0;
    padding: 0.4rem 0.7rem;
    overflow-wrap: anywhere;
    border: 1px solid rgba(117, 169, 255, 0.28);
    border-radius: 10px;
    color: var(--dc-highlight);
    background: rgba(75, 141, 255, 0.09);
}
.dc-status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.55rem;
    border: 1px solid var(--dc-border);
    border-radius: 999px;
    color: var(--dc-muted);
    background: rgba(12, 22, 39, 0.8);
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
.dc-thinking {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    color: var(--dc-muted);
}
.dc-thinking-dots {
    display: inline-flex;
    gap: 0.2rem;
}
.dc-thinking-dots i {
    width: 0.35rem;
    height: 0.35rem;
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

@media (max-width: 900px) {
    [data-testid="stAppViewBlockContainer"] {
        padding: 1rem 1.15rem 13rem !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.75rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    [data-testid="stBottom"] > div {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    [data-testid="stChatInput"] {
        max-width: 100% !important;
        width: 100% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([aria-label*="user" i]) {
        max-width: 88% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) {
        max-width: 100% !important;
    }
}
@media (max-width: 480px) {
    [data-testid="stAppViewBlockContainer"] {
        padding-right: 0.85rem !important;
        padding-left: 0.85rem !important;
    }
    .dc-topline {
        align-items: flex-start !important;
        flex-direction: column !important;
        gap: 0.4rem !important;
    }
}"""


def _apply_styles() -> None:
    """Apply the refined DocuChat design system with Nunito typography."""
    st.html(f"<style>\n{CUSTOM_STYLESHEET.strip()}\n</style>")


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
    if "view" in st.query_params and st.query_params["view"] in ("home", "chat", "documents"):
        st.session_state[ACTIVE_VIEW_KEY] = st.query_params["view"]
    if st.query_params.get("demo") == "chat" and not st.session_state[SESSION_MESSAGES_KEY]:
        st.session_state[ACTIVE_VIEW_KEY] = "chat"
        demo_turn = append_user_message(
            st.session_state,
            "What are the core guidelines for document processing and security in DocuChat AI?",
        )
        if demo_turn:
            append_assistant_message(
                st.session_state,
                demo_turn,
                "Here are the core guidelines and architecture principles for DocuChat AI:\n\n"
                "1. **Session Isolation & Privacy**: All uploaded documents, extracted text chunks, and vector indexes remain strictly isolated in-memory within your active browser session. No data is persisted across browser reloads or shared with other sessions.\n\n"
                "2. **Document Ingestion & File Formats**: Supports PDF and UTF-8 encoded TXT files up to 10 MB each, with a maximum of 5 files per session. PDFs are validated for valid header structures and non-scanned textual content.\n\n"
                "3. **Semantic Embedding & Retrieval**: Documents are parsed into clean text chunks (up to 250 chunks per file) and indexed using Gemini embeddings. Relevant passages are dynamically retrieved using cosine similarity scoring.\n\n"
                "4. **Grounded Answers & Citations**: Answers are generated with temperature 0.2 to minimize hallucinations, and every retrieved passage cites its exact source file for verification.",
                citations=("DocuChat-Architecture.pdf", "Security-Whitepaper.txt"),
            )
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

    content_column, context_column = st.columns([0.68, 0.32], gap="large")
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

    submitted_prompt = None
    if view != "documents":
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
            <p>Explore ideas, understand documents, and get meaningful answers with DocuChat AI.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dc-section-title">Start with an idea</div>', unsafe_allow_html=True)
    selected_prompt = None
    columns = st.columns(3, gap="small")
    home_suggestions = (
        ("Study Notes", "Help me study. Ask which subject or notes I want to work on."),
        ("Summarize Documents", "Summarize the key points from my uploaded documents."),
        ("Explain a Topic", "Help me understand a topic clearly. Ask which topic I have in mind."),
    )
    for column, (label, prompt) in zip(columns, home_suggestions):
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
        st.markdown('<div class="dc-chat-spacer"></div>', unsafe_allow_html=True)
        return None

    st.markdown('<div class="dc-section-title">Try a conversation starter</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        if st.button(
            "Explain a topic",
            key="chat_suggestion_Explain a Topic",
            disabled=not api_key_configured,
            use_container_width=True,
        ):
            return "Help me understand a topic clearly. Ask which topic I have in mind."
    with col2:
        if st.button(
            "Summarize documents",
            key="chat_suggestion_Summarize Documents",
            disabled=not api_key_configured,
            use_container_width=True,
        ):
            if not st.session_state[rag_integration.DOCUMENTS_KEY]:
                st.session_state[ACTIVE_VIEW_KEY] = "documents"
                st.rerun()
            return "Summarize the key points from my uploaded documents."
    with col3:
        if st.button(
            "Brainstorm ideas",
            key="chat_suggestion_Brainstorm Ideas",
            disabled=not api_key_configured,
            use_container_width=True,
        ):
            return "Help me brainstorm ideas for a project and ask about my goals if needed."
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
                    f'<div class="dc-document-title" title="{escape(status["filename"])}">'
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
        '<div class="dc-topline"><h2>Your documents</h2>'
        f'<span class="dc-status-pill">{len(documents)} indexed · {chunks} chunks</span></div>',
        unsafe_allow_html=True,
    )
    _render_document_panel(settings, api_key_configured)


def _render_document_panel(
    settings: config.Settings, api_key_configured: bool
) -> None:
    """Render upload, indexing, status, removal, and clear controls."""
    documents = st.session_state[rag_integration.DOCUMENTS_KEY]
    statuses = st.session_state[rag_integration.DOCUMENT_STATUSES_KEY]
    st.session_state.setdefault(UPLOAD_WIDGET_VERSION_KEY, 0)
    with st.container(border=True):
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
                row, remove = st.columns([3, 2], vertical_alignment="center")
                with row:
                    status_class = {
                        "Indexed": "indexed",
                        "Failed": "failed",
                        "Indexing": "indexing",
                    }.get(status["status"], "")
                    st.markdown(
                        f'<div class="dc-document-title" title="{escape(status["filename"])}">'
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
                st.markdown(
                    f'<div class="dc-document-title" title="{escape(uploaded_file.name)}">'
                    f'{escape(uploaded_file.name)} · <span class="dc-muted">{state}</span></div>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
