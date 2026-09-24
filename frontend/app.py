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

st.set_page_config(page_title="Architecture evolution RAG v.0.1", layout="wide")

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
            # Bumped every time a fresh batch of pending questions is stored (see the
            # `ask_result["questions"]` branch below). Each question row's widget key includes
            # this round number. Without it, round 2's row at index 0 reuses round 1's row-0
            # widget key, and Streamlit restores round 1's stale typed answer (or its Irrelevant/
            # Infer/Suggest selection) into round 2's, DIFFERENT, question — it looks
            # "pre-answered" even though the reviewer never touched it this round.
            round_key = f"ask_more_round::{source}"
            # "Publish" calls `finalize_document`, which persists a SilverDocument version AND
            # runs Gold extraction — the only non-idempotent-by-click action on this whole card.
            # Every OTHER action here (Regenerate, Ask me more, Submit answers) already follows
            # the two-phase "set an in-progress flag, `st.rerun()`, THEN do the real work on the
            # next run" pattern below, which disables its own button the instant it is clicked.
            # Publish used to do its work inline, in the same run as the click, with no such
            # flag — so a genuine double-click (two widget-trigger events queued by the browser
            # before Streamlit re-rendered the button as disabled) could fire two overlapping
            # `POST /transcriptions/finalize` requests. Both can pass `extract_and_persist_
            # gold_facts`'s `already_extracted` check before either commits (that check has no
            # row lock), so both run their own non-deterministic LLM extraction — producing a
            # spurious version-2 in `gold_evolution` for any entity whose two independent
            # extractions did not phrase the narrative byte-for-byte identically. This flag
            # closes the client-side half of that race; `agents/stages/gold/service.py`'s
            # `extract_and_persist_gold_facts` closes the other half with a Postgres advisory
            # lock, for any caller that is not this button (two browser tabs, a retried request).
            publish_key = f"publish_in_progress::{source}"

            regenerating = st.session_state.setdefault(regen_key, False)
            asking_more = st.session_state.setdefault(ask_more_key, False)
            publishing = st.session_state.setdefault(publish_key, False)
            busy = regenerating or asking_more or publishing

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
                    except requests.RequestException:
                        st.error("Failed, try again.")
                    finally:
                        st.session_state[publish_key] = False
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
                        # No HTTP response to read a detail from at all — a timeout (e.g. every
                        # LLM provider hanging because the account behind it is out of credits,
                        # a real observed case) or a dropped connection. `finally` below already
                        # clears `regen_key`, so this card's buttons — Regenerate, Ask me more,
                        # AND Publish — are all enabled again on the very next render, ready to
                        # retry once whatever caused the timeout is fixed.
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
        "Transcript (.vtt, .txt, .md)",
        type=["txt", "vtt", "md"],
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
        # This keys each widget by position, not by question text. The backend already
        # de-dupes exact-duplicate questions (agents.graph._top_questions). But keying widgets
        # off arbitrarily long, LLM-generated question text would still be fragile. An index is
        # always unique, and always short.
        #
        # `round_key`/`round_number` are the other half of that key. `thread_id` alone is NOT
        # unique per batch of questions: `route_after_boss` can loop back to `ask_human` a
        # second time within the SAME thread_id (one material-contradiction escalation, see
        # `agents/graph.py`), so a second, different batch can land at the exact same
        # `(thread_id, index)` pair a first batch already used. Without the round number, that
        # second batch's row 0 would reuse the first batch's row-0 widget key, and Streamlit
        # would restore the first batch's stale typed answer — or its Irrelevant/Infer/Suggest
        # selection — into the second batch's unrelated question. See the matching "Ask me
        # more" fix below, which bumps `ask_more_round::{source}` for the same reason.
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
                        # A second (or later) batch of pending questions for the SAME
                        # thread_id — see the `round_key` comment above for why this must bump.
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

    # This is hand-rolled `st.columns` rows, not a single `st.dataframe`. A dataframe cell can
    # only ever be a link (`LinkColumn`), never a genuine `st.download_button` — and a plain
    # `<a>` link to a `data:` URI does not reliably trigger a download across browsers for a
    # text mime type (most just navigate to it and render the raw text instead). "Download" per
    # row needs the real widget, the same one `_adr_viewer_page` already uses for its own single
    # ADR, so this reuses that exact mechanism instead of a second, weaker one.
    #
    # "View ADR" keeps the exact query-param scheme `LinkColumn` used before
    # (`?view_adr=...&view_adr_version=...`, the same scheme `gold_service.current_architecture_diagram`'s
    # own node links and `main()`'s `_adr_viewer_page` branch already read back) — only the
    # widget rendering it changed, from a `LinkColumn` cell to `st.link_button`, which still
    # opens in a new tab. A new tab means a brand new Streamlit session, so it asks for login
    # again — not a defect, just how Streamlit tabs work.
    header_cols = st.columns([2.4, 3, 1, 2, 1.3, 1.3])
    for col, label in zip(header_cols, ["ADR ID", "Source", "Version", "Ingestion date", "", ""]):
        if label:
            col.markdown(f"**{label}**")

    for adr in history["adrs"]:
        id_col, source_col, version_col, date_col, view_col, download_col = st.columns(
            [2.4, 3, 1, 2, 1.3, 1.3]
        )
        row_key = f"{adr['source_component']}-v{adr['version']}"
        # Same "<source_component> v<version>" wording the chat itself gives back when asked
        # which ADR a fact came from (`answer_question`/`answer_evolution_question` in
        # `agents/stages/gold/service.py`) — so a chat answer can be matched against this
        # column by eye, character for character, no format translation needed.
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
    """This is the landing page that a "View ADR" link opens, in its new tab
    (`?view_adr=...&view_adr_version=...`, read by `main()`). It shows a single, read-only
    ADR. It looks up that ADR from the same `architecture_history` payload the table itself
    renders from. There is no separate endpoint for this."""
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

    # Blue metadata header — every field this ADR version carries — followed by the document
    # itself, then the soft-yellow Gold cards for whatever this exact version generated. Both
    # helpers live in `theme.py`, next to `score_bar_html`, following the same "HTML string
    # built in Python, rendered with `unsafe_allow_html`" convention as the rest of this app.
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
    """This is the landing page a Monitor-tab "View prompt" link opens, in its new tab
    (`?view_prompt=...`, read by `main()`) — the same pattern `_adr_viewer_page` already
    established for "View ADR": no dedicated endpoint of its own, just the already-fetched
    `/v1/frontend/test-monitor` payload, searched here for the one row this link points at.
    That payload already carries the full, untruncated `prompt` text — the Monitor table only
    ever drops it to keep each row one line tall."""
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


def _chat_tab(username: str) -> None:
    title_col, clear_col = st.columns([5, 1])
    with title_col:
        st.subheader("Chat with RAG")
        st.caption("Once an ADR is accepted, ask about the architecture and the timeline of its components.")
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    history = st.session_state.setdefault("chat_history", [])
    for role, text in history:
        with st.chat_message(role):
            st.write(text)

    question = st.chat_input("Ask about the architecture's evolution")
    if question:
        prior_turns = list(history)
        history.append(("user", question))
        with st.spinner("Thinking..."):
            result = api_client.chat(username, question, history=prior_turns)
        history.append(("assistant", result["answer"]))
        st.rerun()


def _format_when(iso_timestamp: str) -> str:
    """`created_at` arrives as a full ISO-8601 string (with seconds, microseconds, and a UTC
    offset) — the Monitor table only ever needs to show it as `yyyy-MM-dd HH:mm`."""
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
        # Hand-rolled `st.columns` rows, not `st.dataframe` — same reason as Architecture
        # history's own table: a dataframe cell can only ever be a link, never a real button,
        # and "View prompt" needs one. Dropping the raw prompt text from this table entirely
        # (instead of a truncated dataframe cell) is the actual fix for the column being
        # unreadable; "View prompt" is how you get the untruncated text back, in a new tab, off the SAME
        # `/v1/frontend/test-monitor` payload this tab already fetched — no second endpoint
        # just to look up one prompt by id.
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

        # Sums only the rows actually shown above — `test_monitor` caps at
        # `_LLM_COSTS_DISPLAY_LIMIT` (200) most recent rows, so this is not necessarily this
        # tenant's all-time total once it has made more calls than that. The label says so
        # explicitly instead of implying a lifetime total it cannot actually back.
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

    # Same new-tab scheme as "View ADR" above, for the Monitor tab's own "View prompt" button
    # (`?view_prompt=<llm_costs.id>`, read back by `_prompt_viewer_page`).
    prompt_id = st.query_params.get("view_prompt")
    if prompt_id:
        _prompt_viewer_page(st.session_state.user["username"], int(prompt_id))
        return

    _main_app()


if __name__ == "__main__":
    main()
