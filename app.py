"""DocuChat AI Streamlit chat application."""

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
MAX_UPLOAD_FILES = 5
MAX_UPLOAD_FILE_SIZE_MB = 10
SUGGESTIONS = (
    ("Explain a topic", "Explain how neural networks learn, in simple terms."),
    ("Help me brainstorm", "Help me brainstorm a few creative project ideas."),
    ("Make a plan", "Help me turn a big goal into a practical step-by-step plan."),
)


def _apply_styles() -> None:
    """Apply the local system-font theme and responsive chat styling."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8fc;
            color: #20243a;
            font-family: "Nunito", "Segoe UI", Arial, sans-serif;
        }
        [data-testid="stSidebar"] {
            background: #191c31;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f3f4ff;
        }
        [data-testid="stSidebar"] button {
            border-radius: 10px;
            min-height: 42px;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid #e7e9f4;
            border-radius: 16px;
            padding: 0.8rem 1rem;
            background: #fff;
        }
        [data-testid="stChatInput"] {
            border-color: #d9ddf1;
            border-radius: 16px;
        }
        .docuchat-eyebrow {
            color: #6670cb;
            font-size: 0.875rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .docuchat-subtitle {
            color: #62677c;
            font-size: 1rem;
            margin-top: -0.5rem;
        }
        @media (max-width: 640px) {
            [data-testid="stChatMessage"] {
                padding: 0.65rem 0.75rem;
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


def _handle_prompt(prompt: str, settings: config.Settings) -> None:
    """Store a prompt, stream its response, and save only completed turns."""
    turn_id = append_user_message(st.session_state, prompt)
    if turn_id is None:
        return

    with st.chat_message("user"):
        st.markdown(prompt.strip())

    with st.chat_message("assistant"):
        progress = st.empty()
        progress.caption("Gemini is responding...")
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
    """Render the chat interface and process one submitted prompt."""
    st.set_page_config(page_title="DocuChat AI", page_icon="💬", layout="wide")
    settings = config.load_settings()
    initialize_conversation_state(st.session_state)
    rag_integration.initialize_document_state(st.session_state)
    _apply_styles()

    api_key_configured = bool(
        settings.gemini_api_key and settings.gemini_api_key.strip()
    )
    messages = st.session_state[SESSION_MESSAGES_KEY]

    with st.sidebar:
        st.markdown("## DocuChat AI")
        st.caption("Your private AI conversation space")
        st.divider()
        st.markdown("### Conversation")
        if st.button("＋  New Chat", key="new_chat", use_container_width=True):
            start_new_conversation(st.session_state)
            st.rerun()
        if st.button(
            "Clear conversation",
            key="clear_conversation",
            disabled=not messages,
            use_container_width=True,
        ):
            clear_conversation(st.session_state)
            st.rerun()

        st.divider()
        if api_key_configured:
            st.success("Gemini ready")
        else:
            st.warning("API key required")
        st.caption(f"Model: `{settings.gemini_chat_model}`")
        st.caption(
            f"{len(get_completed_turns(st.session_state))} completed turns · "
            f"up to {settings.chat_history_limit} retained"
        )
        st.caption("This browser session is isolated from other sessions.")

    _render_document_panel(settings, api_key_configured)

    st.markdown(
        '<div class="docuchat-eyebrow">AI CHAT ASSISTANT</div>',
        unsafe_allow_html=True,
    )
    st.title("DocuChat AI")
    st.markdown(
        '<div class="docuchat-subtitle">A thoughtful space to ask, explore, and create.</div>',
        unsafe_allow_html=True,
    )

    if not api_key_configured:
        st.info("Set `GEMINI_API_KEY` in your local environment to start chatting.")

    for message in messages:
        with st.chat_message(message["role"]):
            if message["is_error"]:
                st.error(message["content"])
            else:
                st.markdown(message["content"])
                if message.get("citations"):
                    st.caption("Document-grounded answer")
                    st.text("Sources: " + " · ".join(message["citations"]))

    suggested_prompt = None
    if not messages:
        st.markdown("### Welcome. What would you like to explore?")
        st.write("Start with a question or choose an idea to get the conversation going.")
        suggestion_columns = st.columns(len(SUGGESTIONS))
        for column, (label, prompt) in zip(suggestion_columns, SUGGESTIONS):
            with column:
                if st.button(
                    label,
                    key=f"suggestion_{label}",
                    disabled=not api_key_configured,
                    use_container_width=True,
                ):
                    suggested_prompt = prompt

    submitted_prompt = st.chat_input(
        "Message DocuChat AI...",
        key="chat_input",
        disabled=not api_key_configured,
    )
    prompt = suggested_prompt or submitted_prompt
    if prompt is not None:
        _handle_prompt(prompt, settings)


def _render_document_panel(
    settings: config.Settings, api_key_configured: bool
) -> None:
    """Render upload, indexing, status, removal, and clear controls."""
    documents = st.session_state[rag_integration.DOCUMENTS_KEY]
    statuses = st.session_state[rag_integration.DOCUMENT_STATUSES_KEY]
    st.session_state.setdefault(UPLOAD_WIDGET_VERSION_KEY, 0)
    with st.expander(f"Documents · {len(documents)} indexed", expanded=False):
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

        if statuses:
            for document_id, status in list(statuses.items()):
                row, remove = st.columns([5, 1])
                with row:
                    st.text(f"{status['filename']} · {status['status']}")
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
