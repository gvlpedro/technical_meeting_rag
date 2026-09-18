"""This is the Streamlit frontend. It matches the "User interface" section of the root
README: one tab per bullet point there. Every tab is thin. It collects input, calls
`api_client`, and renders the response. All the real logic lives in the backend
(`app/routers/frontend.py`): the auth check, the LangGraph run, and Gold retrieval. This file
never touches the database or an LLM directly.

Run:
    uv run streamlit run frontend/app.py

See GETTING_STARTED.md's "Frontend usage" section for the two example logins.
"""

from datetime import date, datetime
from urllib.parse import quote

import requests
import streamlit as st

import api_client
import theme

st.set_page_config(page_title="Technical Meeting RAG", layout="wide")

# These are the three literal answer strings that a clarification-question action button sends,
# instead of typed text. They must match `agents.graph.py`'s own copies exactly:
# `INFER_FROM_CONTEXT_MARKER`, `SUGGEST_INFO_MARKER`, and `"[irrelevant]"` in
# `_DECLINE_PHRASES`. This match matters because the frontend and backend are separate
# processes, with no shared import. The CLARIFICATION ANSWER MARKERS section in
# `prompts/adr_generation/generator.jinja` is what actually interprets the two non-decline
# markers.
_IRRELEVANT_MARKER = "[IRRELEVANT]"
_INFER_MARKER = "[INFER FROM CONTEXT]"
_SUGGEST_MARKER = "[SUGGEST INFO]"


def _render_question_row(question: str, key_prefix: str, disabled: bool) -> str:
    """This renders one clarification question: its text input, plus three action buttons the
    reviewer can click instead of typing. The buttons always show the same colors (red, orange,
    green), using `theme.py`'s own `qbtn-*` CSS. "Irrelevant" means this question does not need
    an answer. "Infer an answer" means the answer is already given elsewhere in what we already
    have, so the system should look it up instead of asking. "Suggest info" explicitly lets the
    ADR propose an answer; that answer is always labeled `LLM SUGGESTION:` in the output.

    This function returns whatever should be sent as this question's answer: the typed text, or
    the matching marker.

    Clicking an already-selected action toggles it back off, so we do not need a separate
    "clear" control. While an action is selected, the text input is disabled. A typed answer and
    a marker are mutually exclusive: they cannot both apply to the same question.

    This renders as two rows, not one. The question takes the full row on its own line, because
    it is often longer than the 60% width an inline label would leave for it. A second row then
    splits 60/40 between the answer box and the three action buttons. Keeping them on the same
    row makes both start at the same vertical position. Otherwise, if the question were used as
    the input's own label, the buttons would sit one line higher than the input."""
    action_key = f"{key_prefix}::action"
    selected = st.session_state.get(action_key)

    st.markdown(f"**{question}**")
    text_col, buttons_col = st.columns([6, 4])
    typed = text_col.text_input(
        question,
        key=f"{key_prefix}::text",
        label_visibility="collapsed",
        disabled=disabled or selected is not None,
    )
    irrelevant_col, infer_col, suggest_col = buttons_col.columns(3)

    if irrelevant_col.button("Irrelevant", key=f"qbtn-irrelevant-{key_prefix}", disabled=disabled, use_container_width=True):
        st.session_state[action_key] = None if selected == "irrelevant" else "irrelevant"
        st.rerun()
    if infer_col.button("Infer an answer", key=f"qbtn-infer-{key_prefix}", disabled=disabled, use_container_width=True):
        st.session_state[action_key] = None if selected == "infer" else "infer"
        st.rerun()
    if suggest_col.button("Suggest info", key=f"qbtn-suggest-{key_prefix}", disabled=disabled, use_container_width=True):
        st.session_state[action_key] = None if selected == "suggest" else "suggest"
        st.rerun()

    if selected == "irrelevant":
        st.caption("Marked irrelevant — will be submitted as not needing an answer.")
        return _IRRELEVANT_MARKER
    if selected == "infer":
        st.caption("Will ask the system to infer this from information already provided, "
                   "instead of leaving it unresolved.")
        return _INFER_MARKER
    if selected == "suggest":
        st.caption("Will let the system suggest new info for this — always labeled "
                   "\"LLM SUGGESTION:\" in the resulting ADR.")
        return _SUGGEST_MARKER
    return typed


def _error_detail(exc: requests.HTTPError) -> str:
    """This returns the backend's own `{"detail": ...}` message, when there is one. But the
    backend does not always get to send that shape. An unhandled exception never reaches a
    FastAPI `HTTPException`. For example, this actually happened once when every real LLM
    provider failed at once (see `llm.router.AllProvidersFailedError`). In that case,
    Starlette's own default handler returns a plain text 500 body instead. Calling
    `response.json()` on that body raises its own `JSONDecodeError`. That used to crash this
    screen, instead of showing the original error at all.

    This function falls back to the raw response text, and then to the exception itself. So it
    can never crash the caller."""
    try:
        return str(exc.response.json().get("detail", exc))
    except ValueError:
        return exc.response.text.strip() or str(exc)


def _login_screen() -> None:
    st.caption("Log in with one of the example accounts in GETTING_STARTED.md.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            st.session_state.user = api_client.login(username, password)
            st.rerun()
        except requests.HTTPError:
            st.error("Invalid username or password.")
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend at {api_client.BACKEND_URL}: {exc}")


def _reset_input_transcription_state(thread_id: str, sources: list[str]) -> None:
    """This clears every session_state key tied to one upload and clarification cycle, once
    every ADR candidate from that cycle has been published. A transcript's lifecycle ends at
    Publish. So nothing about that cycle needs to survive into the next one: not the upload
    result, not its drafts, not its per-question widget state.

    This is called right before the `st.rerun()` that follows a Publish click. That way,
    "Input transcription" renders its blank initial form again on the next run, instead of
    leaving the finalized card on screen forever."""
    st.session_state.pop("last_upload", None)
    st.session_state.pop("upload_payload", None)
    st.session_state.pop("upload_ingestion_date", None)
    st.session_state.pop("upload_max_questions", None)
    # This changes the keys of the date, prompt, uploader, and max-questions widgets (see
    # `_input_transcription_tab`). So they render with fresh, empty defaults, instead of the
    # last typed or uploaded values. Without this, a widget with no explicit key would keep
    # its old value across reruns on its own.
    st.session_state["upload_form_generation"] = st.session_state.get("upload_form_generation", 0) + 1
    candidates = st.session_state.get("adr_candidates", {})
    for source in sources:
        candidates.pop(source, None)
        for prefix in (
            "regen_in_progress",
            "ask_more_in_progress",
            "ask_more_pending",
            "ask_more_no_questions",
            "feedback_pending",
        ):
            st.session_state.pop(f"{prefix}::{source}", None)
        st.session_state.pop(f"feedback::{thread_id}::{source}", None)


def _render_adr_candidates(username: str, result: dict) -> None:
    """This renders one review card per generated ADR. Each card shows the completeness score,
    the unresolved points, the document itself, a feedback box, a "Regenerate ADR" button, and
    a "Publish" button.

    This function tracks each source's current draft in `st.session_state.adr_candidates`,
    separate from `result` itself. Regenerating only updates this local state; nothing is
    persisted until "Publish". So a regenerated draft survives reruns without re-uploading.
    """
    candidates = st.session_state.setdefault("adr_candidates", {})
    for source, document in result["documents"].items():
        if candidates.get(source, {}).get("thread_id") != result["thread_id"]:
            candidates[source] = {
                "document": document,
                "score": result.get("scores", {}).get(source, 0),
                "unresolved_points": result.get("unresolved_points", {}).get(source, []),
                "thread_id": result["thread_id"],
                "finalized": False,
            }

    for source, candidate in candidates.items():
        if candidate["thread_id"] != result["thread_id"]:
            continue  # This candidate is from a different, earlier upload, not this run's own.

        with st.expander(f"Generated ADR — {source}", expanded=True):
            st.markdown(theme.score_bar_html(candidate["score"]), unsafe_allow_html=True)
            if candidate["unresolved_points"]:
                st.warning(
                    "Unresolved points:\n" + "\n".join(f"- {p}" for p in candidate["unresolved_points"])
                )
            st.markdown(candidate["document"])

            if candidate["finalized"]:
                st.success(f"Published — version {candidate['finalized_version']} saved to Gold.")
                continue

            regen_key = f"regen_in_progress::{source}"
            ask_more_key = f"ask_more_in_progress::{source}"
            pending_key = f"ask_more_pending::{source}"
            no_questions_key = f"ask_more_no_questions::{source}"

            regenerating = st.session_state.setdefault(regen_key, False)
            asking_more = st.session_state.setdefault(ask_more_key, False)
            busy = regenerating or asking_more

            if st.session_state.pop(no_questions_key, False):
                st.info("No new questions to ask — the transcript already covers everything found so far.")

            # "Ask me more" replaces the feedback box with a fresh, targeted round of
            # questions. Answering these questions and submitting them feeds that Q&A into the
            # SAME regenerate call that the feedback box below uses. It formats the Q&A as one
            # feedback string. So there is only one Actor/Critic call path to maintain:
            # `api_client.regenerate_document`.
            pending_questions = st.session_state.get(pending_key)
            if pending_questions:
                st.info("Answer these to add more detail, then submit to regenerate the ADR.")
                answers: dict[str, str] = {}
                for index, question in enumerate(pending_questions):
                    answers[question] = _render_question_row(
                        question, key_prefix=f"ask_more_answer::{source}::{index}", disabled=busy
                    )
                if st.button(
                    "Submit answers", key=f"ask_more_submit::{source}", type="primary", disabled=busy
                ):
                    st.session_state[f"feedback_pending::{source}"] = "\n".join(
                        f"Q: {q}\nA: {a}" for q, a in answers.items()
                    )
                    st.session_state[regen_key] = True
                    st.session_state[pending_key] = None
                    st.rerun()
            else:
                # The feedback box and "Regenerate ADR" button are grouped together. The
                # button acts on exactly this text box's content, and nothing else. "Ask me
                # more" and "Publish" are deliberately in their own section below. Neither one
                # submits the feedback box. ("Ask me more" only reads the feedback box, to
                # ground its new questions. See below.)
                feedback_key = f"feedback::{result['thread_id']}::{source}"
                feedback = st.text_area(
                    "Feedback — request changes or details to include",
                    key=feedback_key,
                    disabled=busy,
                )
                if st.button(
                    "Regenerate ADR", key=f"regenerate::{source}", disabled=busy or not feedback
                ):
                    st.session_state[f"feedback_pending::{source}"] = feedback
                    st.session_state[regen_key] = True
                    st.rerun()

                st.markdown("---")
                ask_more_col, launch_col = st.columns(2)
                if ask_more_col.button("Ask me more", key=f"ask_more::{source}", disabled=busy):
                    st.session_state[ask_more_key] = True
                    st.rerun()

                if launch_col.button("Publish", key=f"launch::{source}", type="primary", disabled=busy):
                    try:
                        finalize_result = api_client.finalize_document(username, source, candidate["document"])
                        candidates[source]["finalized"] = True
                        candidates[source]["finalized_version"] = finalize_result["version"]
                        st.toast(
                            f"Published — version {finalize_result['version']} saved to Gold.",
                            icon="✅",
                        )
                        # Only reset once every candidate from THIS upload is published. In a
                        # multi-file batch, other cards may still need review or publishing.
                        # Those cards must keep their state until they are done too.
                        thread_sources = [
                            s for s, c in candidates.items() if c["thread_id"] == result["thread_id"]
                        ]
                        if all(candidates[s]["finalized"] for s in thread_sources):
                            _reset_input_transcription_state(result["thread_id"], thread_sources)
                    except requests.HTTPError as exc:
                        st.error(f"Publish failed: {_error_detail(exc)}")
                    st.rerun()

            if st.session_state[ask_more_key]:
                with st.spinner("Drafting new clarification questions — same question limit as the original upload..."):
                    try:
                        # This is the number the reviewer set on "Input transcription". "Ask
                        # me more" reuses it instead of asking again, as the user requested.
                        max_questions = st.session_state.get("upload_max_questions", 10)
                        # This uses the draft as it stands right now, plus whatever is
                        # currently in the feedback box, submitted or not. Before this, the
                        # code grounded on a stale database copy of the ORIGINAL draft. That
                        # is exactly what made it re-ask questions that were already answered.
                        current_feedback = st.session_state.get(f"feedback::{result['thread_id']}::{source}", "")
                        ask_result = api_client.ask_more_questions(
                            username, source, max_questions, candidate["document"], current_feedback
                        )
                        if ask_result["questions"]:
                            st.session_state[pending_key] = ask_result["questions"]
                        else:
                            st.session_state[no_questions_key] = True
                    except requests.HTTPError as exc:
                        st.error(f"Ask me more failed: {_error_detail(exc)}")
                    finally:
                        st.session_state[ask_more_key] = False
                st.rerun()

            if st.session_state[regen_key]:
                with st.spinner("Regenerating the ADR with your feedback..."):
                    try:
                        regen_result = api_client.regenerate_document(
                            source,
                            st.session_state[f"feedback_pending::{source}"],
                            username,
                            candidate["document"],
                        )
                        candidates[source] = {
                            **candidate,
                            "document": regen_result["document"],
                            "score": regen_result["score"],
                            "unresolved_points": regen_result["unresolved_points"],
                        }
                    except requests.HTTPError as exc:
                        st.error(f"Regenerate failed: {_error_detail(exc)}")
                    finally:
                        st.session_state[regen_key] = False
                st.rerun()


def _input_transcription_tab(username: str) -> None:
    st.subheader("Input transcription")
    st.caption(
        "Type a prompt, upload transcriptions/PDFs, or both, for one meeting, to be processed "
        "and clarified."
    )

    uploading = st.session_state.setdefault("upload_in_progress", False)
    # This changes the key of every initial-form widget. `_reset_input_transcription_state`
    # bumps this value after a full publish. So each widget gets a brand-new key that Streamlit
    # has never seen before, and there is no prior value to restore. Without this, a plain
    # `pop()` of the old key would not help: a widget with no explicit key keeps its own
    # last-typed value across reruns regardless.
    form_gen = st.session_state.setdefault("upload_form_generation", 0)

    ingestion_date = st.date_input(
        "Meeting date", value=date.today(), disabled=uploading, key=f"ingestion_date::{form_gen}"
    )
    prompt_text = st.text_area(
        "Prompt — type the discussion directly instead of uploading a .txt file",
        disabled=uploading,
        help="Treated exactly like an uploaded prompt.txt — useful for a quick test without "
        "creating a file first. Fine to use together with real uploads below.",
        key=f"prompt_text::{form_gen}",
    )
    uploaded = st.file_uploader(
        "Transcript (.vtt, .txt, .md) and extra documentation (.pdf)",
        type=["txt", "vtt", "md", "pdf"],
        accept_multiple_files=True,
        disabled=uploading,
        key=f"uploaded_files::{form_gen}",
    )
    max_questions_per_stage = st.number_input(
        "Max questions per stage (architecture + data contracts)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        disabled=uploading,
        help="Applies the same limit to both stages — 4 here means at most 4 architecture "
        "questions and 4 data-contract questions, 8 in total.",
        key=f"max_questions_per_stage::{form_gen}",
    )

    has_input = bool(uploaded) or bool(prompt_text.strip())
    if st.button("Process and clarify", disabled=uploading or not has_input, type="primary"):
        # This saves the file bytes and this click's own inputs into session_state. Then it
        # reruns once, BEFORE the actual, slow, real-LLM call. This is what lets the button
        # below render as disabled while the call is in flight. Without this extra rerun, the
        # button would only render as disabled AFTER the blocking call already finished. That
        # would leave a window where an impatient extra click starts a second graph run, which
        # would be billed separately.
        payload = [(f.name, f.getvalue()) for f in uploaded] if uploaded else []
        if prompt_text.strip():
            # This uses the same extension the file uploader itself accepts. So the backend
            # never learns that this text came from a text box instead of a real file.
            # (`ingestion.service.ingest_uploaded_files` decodes .txt as plain text either
            # way.) The file name includes a timestamp, instead of a fixed name like
            # "prompt.txt". A fixed name would make every typed prompt, across totally
            # unrelated test sessions, resolve to the SAME source_component. That would
            # silently pool or version-chain content that has nothing to do with each other.
            prompt_filename = f"prompt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.txt"
            payload.append((prompt_filename, prompt_text.encode("utf-8")))
        st.session_state.upload_payload = payload
        st.session_state.upload_ingestion_date = ingestion_date.strftime("%Y%m%d")
        st.session_state.upload_max_questions = int(max_questions_per_stage)
        st.session_state.upload_in_progress = True
        st.rerun()

    if st.session_state.upload_in_progress:
        with st.spinner(
            "Ingesting and running the clarification pipeline — this calls the real LLM "
            "several times and can take a few minutes. Please wait, do not click again."
        ):
            try:
                st.session_state.last_upload = api_client.upload_transcription(
                    st.session_state.upload_ingestion_date,
                    st.session_state.upload_payload,
                    st.session_state.upload_max_questions,
                    username,
                )
            except requests.HTTPError as exc:
                st.error(f"Upload failed: {_error_detail(exc)}")
            finally:
                st.session_state.upload_in_progress = False
        st.rerun()

    result = st.session_state.get("last_upload")
    if not result:
        return

    if result["status"] == "pending_review":
        st.warning(
            "The clarification loop needs a human answer before it can finish. Answer every "
            "question below, then submit."
        )
        resuming = st.session_state.setdefault("resume_in_progress", False)
        answers: dict[str, str] = {}
        # This keys each widget by position, not by question text. The backend already
        # de-dupes exact-duplicate questions (agents.graph._top_questions). But keying widgets
        # off arbitrarily long, LLM-generated question text would still be fragile. An index is
        # always unique, and always short.
        for index, question in enumerate(result["pending_questions"]):
            answers[question] = _render_question_row(
                question, key_prefix=f"answer::{result['thread_id']}::{index}", disabled=resuming
            )

        if st.button("Submit answers", type="primary", disabled=resuming):
            st.session_state.resume_answers = dict(answers)
            st.session_state.resume_thread_id = result["thread_id"]
            st.session_state.resume_in_progress = True
            st.rerun()

        if st.session_state.resume_in_progress:
            with st.spinner(
                "Resuming the clarification loop — this calls the real LLM again and can "
                "take a few minutes. Please wait, do not click again."
            ):
                try:
                    st.session_state.last_upload = api_client.resume_transcription(
                        st.session_state.resume_thread_id, st.session_state.resume_answers, username
                    )
                except requests.HTTPError as exc:
                    st.error(f"Resume failed: {_error_detail(exc)}")
                finally:
                    st.session_state.resume_in_progress = False
            st.rerun()
    else:
        st.success(f"Done. Files ingested: {', '.join(result['files_ingested']) or '(none — already ingested)'}")
        _render_adr_candidates(username, result)


def _architecture_history_tab(username: str) -> None:
    st.subheader("Architecture history")
    st.caption(
        "Current architecture, built live from Gold's own component state, and every ADR "
        "published so far."
    )

    history = api_client.architecture_history(username)

    if history["diagram"]:
        st.mermaid_chart(history["diagram"])
        # If the rendered diagram above ever looks wrong, for example boxes missing or only
        # edges visible, this is the exact Mermaid source it was given. Compare the two before
        # reporting a bug. This tells apart a generation bug, where the source itself is wrong,
        # from a rendering bug, where the source is fine but `st.mermaid_chart` does not draw
        # it.
        with st.expander("View Mermaid source"):
            st.code(history["diagram"], language="text")
    else:
        st.info("No components in Gold yet — publish an ADR from Input transcription to see it here.")

    st.write("#### ADRs")
    if not history["adrs"]:
        st.info("No ADRs published yet.")
        return

    # This uses a real `st.dataframe`, not hand-rolled `st.columns` rows. That way, "View ADR"
    # can be a genuine link cell (`LinkColumn`) that opens in a new browser tab. It uses the
    # same query-param scheme as `gold_service.current_architecture_diagram`'s own node links.
    # `main()`'s `_adr_viewer_page` branch reads that scheme back. Opening that link starts a
    # brand new Streamlit session, because a separate browser tab always does. So it asks for
    # login again. This is not a defect. It is just how Streamlit tabs work.
    rows = [
        {
            "Source": adr["source_component"],
            "Version": adr["version"],
            "Ingestion date": adr["ingestion_date"],
            "View": f"?view_adr={quote(adr['source_component'], safe='')}&view_adr_version={adr['version']}",
        }
        for adr in history["adrs"]
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={"View": st.column_config.LinkColumn(display_text="View ADR")},
    )


def _adr_viewer_page(username: str, source_component: str, version: int) -> None:
    """This is the landing page that a "View ADR" link opens, in its new tab
    (`?view_adr=...&view_adr_version=...`, read by `main()`). It shows a single, read-only
    ADR. It looks up that ADR from the same `architecture_history` payload the table itself
    renders from. There is no separate endpoint for this."""
    st.caption(f"Architecture history — {source_component} v{version}")
    history = api_client.architecture_history(username)
    match = next(
        (
            adr
            for adr in history["adrs"]
            if adr["source_component"] == source_component and adr["version"] == version
        ),
        None,
    )
    if match is None:
        st.error(f"No ADR found for {source_component!r} version {version}.")
        return
    st.markdown(match["content"])


def _chat_tab(username: str) -> None:
    st.subheader("Chat with RAG")
    st.caption("Once an ADR is accepted, ask about the architecture and the timeline of its components.")

    history = st.session_state.setdefault("chat_history", [])
    for role, text in history:
        with st.chat_message(role):
            st.write(text)

    question = st.chat_input("Ask about the architecture's evolution")
    if question:
        history.append(("user", question))
        with st.spinner("Thinking..."):
            result = api_client.chat(username, question)
        history.append(("assistant", result["answer"]))
        st.rerun()


def _test_monitor_tab() -> None:
    st.subheader("Monitor")
    st.caption("Test suite results and LLM token/cost consumption for the whole application.")

    try:
        data = api_client.test_monitor()
    except requests.HTTPError:
        st.info("Monitor is disabled (start_test_mode is off).")
        return

    st.write("#### LLM usage by tenant")
    usage = data["llm_usage_by_tenant"]
    if usage:
        st.table(
            [
                {"tenant": tenant, **bucket}
                for tenant, bucket in usage.items()
            ]
        )
    else:
        st.info("No real LLM calls have been logged yet.")

    st.write("#### Test suite results")
    for suite in data["suites"]:
        with st.expander(suite["suite"]):
            st.json(suite["results"])


def _main_app() -> None:
    user = st.session_state.user

    nav_items = ["Input transcription", "Architecture history", "Chat with RAG"]
    if st.session_state.get("start_test_mode", False):
        nav_items.append("Monitor")
    active_page = st.session_state.setdefault("active_page", nav_items[0])
    if active_page not in nav_items:  # e.g. Monitor got disabled mid-session
        active_page = nav_items[0]

    with st.sidebar:
        st.markdown(f"**{user['username']}**")
        st.caption(f"tenant: {user['tenant']}")
        st.markdown("---")
        # This is the vertical nav: each item is a full-width button. The CURRENT page renders
        # with type="primary". This is only a hook for theme.py's CSS, to style it as "active"
        # (filled, bright text), versus every other item's plain "inactive" look. It never
        # triggers theme.py's own purple primary-button styling, because that styling is
        # scoped to the main content area only, not the sidebar.
        for item in nav_items:
            if st.button(
                item,
                key=f"nav::{item}",
                use_container_width=True,
                type="primary" if item == active_page else "secondary",
            ):
                st.session_state.active_page = item
                st.rerun()
        st.markdown("---")
        if st.button("Log out", use_container_width=True):
            del st.session_state.user
            st.rerun()

    if active_page == "Input transcription":
        _input_transcription_tab(user["username"])
    elif active_page == "Architecture history":
        _architecture_history_tab(user["username"])
    elif active_page == "Chat with RAG":
        _chat_tab(user["username"])
    elif active_page == "Monitor":
        _test_monitor_tab()


def main() -> None:
    theme.inject()

    if "user" not in st.session_state:
        _login_screen()
        return

    if "start_test_mode" not in st.session_state:
        try:
            st.session_state.start_test_mode = api_client.get_config()["start_test_mode"]
        except requests.RequestException:
            st.session_state.start_test_mode = False

    # A "View ADR" link always opens in a NEW browser tab with these query params. This link
    # can come from `_architecture_history_tab`'s table, or from a node in
    # `gold_service.current_architecture_diagram`. A new tab means a fresh Streamlit session,
    # so the login gate above still applies first. Once logged in, this branch takes over the
    # whole page, instead of the normal sidebar nav.
    source_component = st.query_params.get("view_adr")
    version = st.query_params.get("view_adr_version")
    if source_component and version:
        _adr_viewer_page(st.session_state.user["username"], source_component, int(version))
        return

    _main_app()


if __name__ == "__main__":
    main()
