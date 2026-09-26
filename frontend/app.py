"""Streamlit frontend: each tab collects input, calls `api_client`, and renders the response,
while all real logic (auth, the LangGraph run, Gold retrieval) lives in the backend at
`app/routers/frontend.py`.

Run:
    uv run streamlit run frontend/app.py

See GETTING_STARTED.md's "Frontend usage" section for the two example logins.
"""

import html
import re
from datetime import date, datetime
from urllib.parse import quote

import requests
import streamlit as st

import api_client
import theme

st.set_page_config(page_title="Architecture evolution RAG v.0.1", layout="wide")

# The three literal marker strings a clarification action button sends instead of typed
# text — they must match `agents/graph.py`'s own `INFER_FROM_CONTEXT_MARKER`,
# `SUGGEST_INFO_MARKER`, and `_DECLINE_PHRASES` copies exactly, since frontend and backend
# are separate processes with no shared import.
_IRRELEVANT_MARKER = "[IRRELEVANT]"
_INFER_MARKER = "[INFER FROM CONTEXT]"
_SUGGEST_MARKER = "[SUGGEST INFO]"


def _render_question_row(question: str, key_prefix: str, disabled: bool) -> str:
    """Renders one clarification question as a text input plus Irrelevant/Infer an
    answer/Suggest info buttons (mutually exclusive with typed text, each toggling back off
    on a second click), and returns whichever the reviewer chose — the typed text or the
    matching marker."""
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
    """Returns the backend's `{"detail": ...}` message when present, falling back to the raw
    response text and then to the exception itself, since an unhandled backend exception
    (e.g. every LLM provider failing at once, see `llm.router.AllProvidersFailedError`)
    never reaches a structured FastAPI `HTTPException`."""
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
    """Clears every session_state key tied to one upload/clarification cycle once all of its
    ADR candidates are published, called right before the rerun that follows a Publish click
    so "Input transcription" renders a blank form again instead of the finalized card."""
    st.session_state.pop("last_upload", None)
    st.session_state.pop("upload_payload", None)
    st.session_state.pop("upload_ingestion_date", None)
    st.session_state.pop("upload_max_questions", None)
    # Bumps the widget-key generation so the date, prompt, uploader, and max-questions
    # widgets in `_input_transcription_tab` render with fresh, empty defaults instead of
    # keeping their last value across reruns.
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
    """Renders one review card per generated ADR (score, unresolved points, document,
    feedback box, Regenerate/Publish buttons), tracking each source's current draft in
    `st.session_state.adr_candidates` so a regenerated draft survives reruns without
    persisting anything until Publish."""
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
            # Bumped each time a fresh batch of pending questions is stored, so a new round's
            # widget keys don't collide with a prior round's and Streamlit doesn't restore a
            # stale answer into a different question.
            round_key = f"ask_more_round::{source}"
            # Publish persists a SilverDocument version and runs Gold extraction — the only
            # non-idempotent action on this card — so, unlike every other button here, it
            # needs this in-progress flag to stop a genuine double-click from firing two
            # overlapping finalize requests; the other half of that race is closed
            # server-side by a Postgres advisory lock in `agents/stages/gold/service.py`.
            publish_key = f"publish_in_progress::{source}"

            regenerating = st.session_state.setdefault(regen_key, False)
            asking_more = st.session_state.setdefault(ask_more_key, False)
            publishing = st.session_state.setdefault(publish_key, False)
            busy = regenerating or asking_more or publishing

            if st.session_state.pop(no_questions_key, False):
                st.info("No new questions to ask — the transcript already covers everything found so far.")

            # "Ask me more" replaces the feedback box with a targeted round of questions
            # whose answers get formatted into one feedback string and fed through the same
            # `api_client.regenerate_document` call, keeping a single Actor/Critic call path.
            pending_questions = st.session_state.get(pending_key)
            if pending_questions:
                st.info("Answer these to add more detail, then submit to regenerate the ADR.")
                answers: dict[str, str] = {}
                round_number = st.session_state.get(round_key, 0)
                for index, question in enumerate(pending_questions):
                    answers[question] = _render_question_row(
                        question, key_prefix=f"ask_more_answer::{source}::{round_number}::{index}", disabled=busy
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
                # The feedback box and "Regenerate ADR" button are grouped together because
                # the button acts only on this text box's content; "Ask me more" and
                # "Publish", below, never submit it (though "Ask me more" reads it to ground
                # its new questions).
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
                    st.session_state[publish_key] = True
                    st.rerun()

            if st.session_state[publish_key]:
                with st.spinner("Publishing — writing to Silver and Gold. Please wait, do not click again."):
                    try:
                        finalize_result = api_client.finalize_document(username, source, candidate["document"])
                        candidates[source]["finalized"] = True
                        candidates[source]["finalized_version"] = finalize_result["version"]
                        st.toast(
                            f"Published — version {finalize_result['version']} saved to Gold.",
                            icon="✅",
                        )
                        # Only resets once every candidate from this upload is published,
                        # since a multi-file batch's other cards may still need review.
                        thread_sources = [
                            s for s, c in candidates.items() if c["thread_id"] == result["thread_id"]
                        ]
                        if all(candidates[s]["finalized"] for s in thread_sources):
                            _reset_input_transcription_state(result["thread_id"], thread_sources)
                    except requests.HTTPError as exc:
                        st.error(f"Publish failed: {_error_detail(exc)}")
                    except requests.RequestException:
                        st.error("Failed, try again.")
                    finally:
                        st.session_state[publish_key] = False
                st.rerun()

            if st.session_state[ask_more_key]:
                with st.spinner("Drafting new clarification questions — same question limit as the original upload..."):
                    try:
                        # Reuses the question limit the reviewer already set on "Input
                        # transcription" instead of asking again.
                        max_questions = st.session_state.get("upload_max_questions", 10)
                        # Grounds on the draft as it stands now plus whatever is in the
                        # feedback box, since grounding on the original database draft used
                        # to re-ask questions already answered.
                        current_feedback = st.session_state.get(f"feedback::{result['thread_id']}::{source}", "")
                        ask_result = api_client.ask_more_questions(
                            username, source, max_questions, candidate["document"], current_feedback
                        )
                        if ask_result["questions"]:
                            st.session_state[pending_key] = ask_result["questions"]
                            st.session_state[round_key] = st.session_state.get(round_key, 0) + 1
                        else:
                            st.session_state[no_questions_key] = True
                    except requests.HTTPError as exc:
                        st.error(f"Ask me more failed: {_error_detail(exc)}")
                    except requests.RequestException:
                        st.error("Failed, try again.")
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
                    except requests.RequestException:
                        # No HTTP response to read here — a timeout (every LLM provider
                        # hanging, an actually observed case) or a dropped connection — but
                        # the `finally` below still clears `regen_key`, re-enabling every
                        # button on this card for a retry.
                        st.error("Failed, try again.")
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
    # Changes the key of every initial-form widget; `_reset_input_transcription_state` bumps
    # it after a full publish so each widget gets a brand-new key with no prior value to
    # restore.
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
        "Transcript (.txt, .md)",
        type=["txt", "md"],
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
        # Saves the inputs into session_state and reruns once before the slow, real-LLM
        # call, so the button below renders disabled while it's in flight instead of leaving
        # a window for an impatient double-click to start a second, separately-billed graph
        # run.
        payload = [(f.name, f.getvalue()) for f in uploaded] if uploaded else []
        if prompt_text.strip():
            # Uses a `.txt` name with a timestamp, not a fixed "prompt.txt", so the backend
            # treats it like a real upload while distinct typed prompts don't collide on the
            # same source_component.
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
            except requests.RequestException:
                st.error("Failed, try again.")
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
        # Keys each widget by position rather than by the arbitrarily long, LLM-generated
        # question text, which would be a fragile key even after the backend's own de-dup in
        # `agents.graph._top_questions`.
        #
        # `round_key`/`round_number` cover the other half of that key, since `thread_id`
        # alone isn't unique per batch — `route_after_boss` can loop back to `ask_human` a
        # second time within the same thread_id — so without it a second batch's row 0 would
        # restore the first batch's stale answer into an unrelated question.
        round_key = f"resume_round::{result['thread_id']}"
        round_number = st.session_state.get(round_key, 0)
        for index, question in enumerate(result["pending_questions"]):
            answers[question] = _render_question_row(
                question, key_prefix=f"answer::{result['thread_id']}::{round_number}::{index}", disabled=resuming
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
                    resume_result = api_client.resume_transcription(
                        st.session_state.resume_thread_id, st.session_state.resume_answers, username
                    )
                    st.session_state.last_upload = resume_result
                    if resume_result["status"] == "pending_review":
                        # A second (or later) batch of pending questions for the same
                        # thread_id, so this must bump — see the `round_key` comment above.
                        resume_round_key = f"resume_round::{resume_result['thread_id']}"
                        st.session_state[resume_round_key] = st.session_state.get(resume_round_key, 0) + 1
                except requests.HTTPError as exc:
                    st.error(f"Resume failed: {_error_detail(exc)}")
                except requests.RequestException:
                    st.error("Failed, try again.")
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
        # Shows the exact Mermaid source behind the diagram above, to tell apart a
        # generation bug (bad source) from a rendering bug (`st.mermaid_chart` not drawing
        # good source).
        with st.expander("View Mermaid source"):
            st.code(history["diagram"], language="text")
    else:
        st.info("No components in Gold yet — publish an ADR from Input transcription to see it here.")

    st.write("#### ADRs")
    if not history["adrs"]:
        st.info("No ADRs published yet.")
        return

    # Hand-rolled `st.columns` rows, not `st.dataframe`, because a dataframe cell can only
    # ever be a link, never a real `st.download_button` — so "Download" reuses the same
    # widget `_adr_viewer_page` already uses for its own ADR.
    #
    # "View ADR" keeps the same `?view_adr=...&view_adr_version=...` query-param scheme
    # `gold_service.current_architecture_diagram`'s node links and `_adr_viewer_page` already
    # read, just rendered with `st.link_button` instead of a `LinkColumn` cell.
    header_cols = st.columns([2.4, 3, 1, 2, 1.3, 1.3])
    for col, label in zip(header_cols, ["ADR ID", "Source", "Version", "Ingestion date", "", ""]):
        if label:
            col.markdown(f"**{label}**")

    for adr in history["adrs"]:
        id_col, source_col, version_col, date_col, view_col, download_col = st.columns(
            [2.4, 3, 1, 2, 1.3, 1.3]
        )
        row_key = f"{adr['source_component']}-v{adr['version']}"
        # Uses the same "<source_component> v<version>" wording the chat gives back for a
        # fact's source (`agents/stages/gold/service.py`), so a chat answer matches this
        # column character for character.
        id_col.code(f"{adr['source_component']} v{adr['version']}", language=None)
        source_col.write(adr["source_component"])
        version_col.write(adr["version"])
        date_col.write(adr["ingestion_date"])
        view_url = f"?view_adr={quote(adr['source_component'], safe='')}&view_adr_version={adr['version']}"
        view_col.link_button("View ADR", view_url, key=f"view-{row_key}", use_container_width=True)
        download_col.download_button(
            "Download",
            data=adr["content"],
            file_name=f"{row_key}.md",
            mime="text/markdown",
            key=f"download-{row_key}",
            use_container_width=True,
        )


def _adr_viewer_page(username: str, source_component: str, version: int) -> None:
    """The landing page a "View ADR" link opens in a new tab
    (`?view_adr=...&view_adr_version=...`, read by `main()`), showing a single read-only ADR
    looked up from the same `architecture_history` payload the table itself uses."""
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
        st.caption(f"Architecture history — {source_component} v{version}")
        st.error(f"No ADR found for {source_component!r} version {version}.")
        return

    # The caption and the download button share one row, button on the right, so the reader
    # can grab the raw markdown without scrolling past the whole document first.
    caption_col, download_col = st.columns([5, 1])
    with caption_col:
        st.caption(f"Architecture history — {source_component} v{version}")
    with download_col:
        st.download_button(
            "⬇️ Markdown",
            data=match["content"],
            file_name=f"{source_component}-v{version}.md",
            mime="text/markdown",
        )

    # Blue metadata header, then the document, then the soft-yellow Gold cards for this
    # version, both rendered via `theme.py` helpers following the same
    # HTML-string-plus-`unsafe_allow_html` convention as the rest of this app.
    st.markdown(
        theme.adr_metadata_header_html(
            source_component=match["source_component"],
            version=match["version"],
            ingestion_date=match["ingestion_date"],
            authored_by=match["authored_by"],
            created_at=match["created_at"],
            content_hash=match["content_hash"],
        ),
        unsafe_allow_html=True,
    )
    st.markdown(match["content"])
    st.markdown("---")
    st.markdown(theme.gold_entity_cards_html(match["gold_entities"]), unsafe_allow_html=True)


def _prompt_viewer_page(username: str, prompt_id: int) -> None:
    """The landing page a Monitor-tab "View prompt" link opens in a new tab
    (`?view_prompt=...`, read by `main()`), showing the full, untruncated prompt text for one
    row of the already-fetched `/v1/frontend/test-monitor` payload."""
    try:
        data = api_client.test_monitor(username)
    except requests.HTTPError:
        st.error("Monitor is disabled (start_test_mode is off).")
        return

    match = next((row for row in data["llm_costs"] if row["id"] == prompt_id), None)
    if match is None:
        st.caption(f"LLM call #{prompt_id}")
        st.error(f"No LLM call found with id {prompt_id} for this tenant.")
        return

    st.caption(
        f"LLM call #{prompt_id} — {match['method']} — {match['model']} — "
        f"{_format_when(match['created_at'])}"
    )
    st.text_area("Prompt", match["prompt"], height=600, disabled=True)


# Matches the exact "<source_component> vN" wording chat answers already use for a fact's
# source (`agents/stages/gold/service.py`'s `build_context_lines`/`_evolution_step_line`), the
# same wording `_architecture_history_tab`'s "View ADR" column already assumes verbatim. The
# required dotted extension (`.txt`/`.md`, plus `.vtt` for ADRs uploaded before that format was
# retired) is what keeps this from matching an unrelated "vN" elsewhere in the prose.
_ADR_MENTION_PATTERN = re.compile(r"\b([\w][\w.\-]*\.(?:txt|md|vtt))\s+v(\d+)\b")


def _linkify_adr_mentions(text: str) -> str:
    """Turns every ADR mention in a chat answer into a link that opens that exact ADR in a new
    tab — the same `?view_adr=...&view_adr_version=...` scheme `_architecture_history_tab`'s own
    "View ADR" button uses (see `main()`'s own comment on why that always opens a fresh tab).

    `text` is escaped FIRST, and the `<a>` tags are only ever built from that already-escaped,
    regex-matched substring — never from the raw LLM output directly — so this stays safe to
    render with `unsafe_allow_html=True` even if a transcript or answer happened to contain
    literal HTML."""
    escaped = html.escape(text)

    def _replace(match: re.Match) -> str:
        source_component, version = match.group(1), match.group(2)
        url = f"?view_adr={quote(source_component, safe='')}&view_adr_version={version}"
        return f'<a href="{url}" target="_blank">{source_component} v{version}</a>'

    return _ADR_MENTION_PATTERN.sub(_replace, escaped)


def _render_relationship_diagram(diagram: str, sources: list[dict]) -> None:
    """Renders `ChatResponse.diagram` (the cited component(s) plus their direct neighbors —
    `gold.build_relationship_diagram`) boxed under the answer, with a small caption underneath
    linking to the ADR(s) the cited component(s) actually came from — same
    `?view_adr=...&view_adr_version=...` new-tab scheme as `_linkify_adr_mentions` and
    `_architecture_history_tab`'s own "View ADR" button. Never a neighbor's ADR: `sources` only
    ever names the entities the answer actually cited (`diagram_sources`)."""
    with st.container(border=True):
        st.mermaid_chart(diagram)
        links = []
        for source in sources:
            url = (
                f"?view_adr={quote(source['source_component'], safe='')}"
                f"&view_adr_version={source['source_adr_version']}"
            )
            label = html.escape(f"{source['canonical_name']} — {source['source_component']} v{source['source_adr_version']}")
            links.append(f'<a href="{url}" target="_blank">{label}</a>')
        if links:
            st.markdown(f"<small>ADR: {' · '.join(links)}</small>", unsafe_allow_html=True)


def _chat_tab(username: str) -> None:
    title_col, clear_col = st.columns([5, 1])
    with title_col:
        st.subheader("Chat with RAG")
        st.caption("Once an ADR is accepted, ask about the architecture and the timeline of its components.")
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_diagrams = {}
            st.rerun()

    history = st.session_state.setdefault("chat_history", [])
    diagrams = st.session_state.setdefault("chat_diagrams", {})
    for i, (role, text) in enumerate(history):
        with st.chat_message(role):
            st.markdown(_linkify_adr_mentions(text), unsafe_allow_html=True)
            if i in diagrams:
                _render_relationship_diagram(diagrams[i]["diagram"], diagrams[i]["sources"])

    question = st.chat_input("Ask about the architecture's evolution")
    if question:
        prior_turns = list(history)
        history.append(("user", question))
        with st.spinner("Thinking..."):
            result = api_client.chat(username, question, history=prior_turns)
        history.append(("assistant", result["answer"]))
        if result.get("diagram"):
            diagrams[len(history) - 1] = {"diagram": result["diagram"], "sources": result.get("diagram_sources", [])}
        st.rerun()


def _format_when(iso_timestamp: str) -> str:
    """Formats a full ISO-8601 `created_at` timestamp down to `yyyy-MM-dd HH:mm` for the
    Monitor table."""
    return datetime.fromisoformat(iso_timestamp).strftime("%Y-%m-%d %H:%M")


def _test_monitor_tab(username: str) -> None:
    st.subheader("Monitor")
    st.caption("Test suite results, and every real LLM call this tenant made and what it cost.")

    try:
        data = api_client.test_monitor(username)
    except requests.HTTPError:
        st.info("Monitor is disabled (start_test_mode is off).")
        return

    st.write("#### LLM calls")
    costs = data["llm_costs"]
    if costs:
        st.caption(f"{len(costs)} most recent call(s) for this tenant, newest first.")
        header_cols = st.columns([1.3, 1.6, 1.4, 0.9, 0.9, 1.6])
        for col, label in zip(header_cols, ["When", "Method", "Model", "Input (€)", "Output (€)", ""]):
            if label:
                col.markdown(f"**{label}**")
        for row in costs:
            when_col, method_col, model_col, input_col, output_col, view_col = st.columns(
                [1.3, 1.6, 1.4, 0.9, 0.9, 1.6]
            )
            when_col.write(_format_when(row["created_at"]))
            method_col.write(row["method"])
            model_col.write(row["model"])
            input_col.write(f"{row['input_cost']:.6f}")
            output_col.write(f"{row['output_cost']:.6f}")
            view_col.link_button(
                "View prompt", f"?view_prompt={row['id']}", key=f"view-prompt-{row['id']}", use_container_width=True
            )

        # Sums only the rows shown above, since `test_monitor` caps at 200 most recent rows
        # and isn't necessarily this tenant's all-time total.
        total_input = sum(row["input_cost"] for row in costs)
        total_output = sum(row["output_cost"] for row in costs)
        when_col, method_col, model_col, input_col, output_col, view_col = st.columns(
            [1.3, 1.6, 1.4, 0.9, 0.9, 1.6]
        )
        method_col.markdown(f"**Total (of the {len(costs)} row(s) shown)**")
        input_col.markdown(f"**{total_input:.6f}**")
        output_col.markdown(f"**{total_output:.6f}**")
    else:
        st.info("No real LLM calls have been logged yet for this tenant.")

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
        # The vertical nav: each item is a full-width button, with the current page's
        # type="primary" only a hook for theme.py's "active" CSS, since that styling is
        # scoped to the main content area, not the sidebar.
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
        _test_monitor_tab(user["username"])


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

    # A "View ADR" link (from `_architecture_history_tab`'s table or a
    # `gold_service.current_architecture_diagram` node) always opens in a new tab, which is a
    # fresh Streamlit session, so the login gate above still applies before this branch takes
    # over the whole page.
    source_component = st.query_params.get("view_adr")
    version = st.query_params.get("view_adr_version")
    if source_component and version:
        _adr_viewer_page(st.session_state.user["username"], source_component, int(version))
        return

    # Same new-tab scheme as "View ADR" above, for the Monitor tab's own "View prompt" button
    # (`?view_prompt=<llm_costs.id>`, read back by `_prompt_viewer_page`).
    prompt_id = st.query_params.get("view_prompt")
    if prompt_id:
        _prompt_viewer_page(st.session_state.user["username"], int(prompt_id))
        return

    _main_app()


if __name__ == "__main__":
    main()
