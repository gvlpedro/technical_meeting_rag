"""Streamlit frontend — the "User interface" section of the root README, one tab per bullet
point there. Every tab is thin: it collects input, calls `api_client`, and renders the
response. All real logic (auth check, the LangGraph run, Gold retrieval) lives in the backend
(`app/routers/frontend.py`) — this file never touches the database or an LLM directly.

Run:
    uv run streamlit run frontend/app.py

See GETTING_STARTED.md's "Frontend usage" section for the two example logins.
"""

from datetime import date
from urllib.parse import quote

import requests
import streamlit as st

import api_client
import theme

st.set_page_config(page_title="Technical Meeting RAG", layout="wide")


def _login_screen() -> None:
    st.title("Technical Meeting RAG")
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


def _render_adr_candidates(tenant: str, result: dict) -> None:
    """Renders one review card per generated ADR: completeness score, unresolved points, the
    document itself, a feedback box, "Regenerate ADR", and "Publish".

    Tracks each source's current draft in `st.session_state.adr_candidates`, independent of
    `result` itself — regenerating updates only this local state (nothing is persisted until
    "Publish"), so a regenerated draft survives reruns without re-uploading.
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
            continue  # a candidate from a different, earlier upload — not this run's own

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

            # "Ask me more" replaces the feedback box with a fresh, targeted question round —
            # answering it and submitting feeds those Q&A into the SAME regenerate call the
            # feedback box below uses, formatted as one feedback string, so there is only one
            # Actor/Critic call path to maintain (`api_client.regenerate_document`).
            pending_questions = st.session_state.get(pending_key)
            if pending_questions:
                st.info("Answer these to add more detail, then submit to regenerate the ADR.")
                answers: dict[str, str] = {}
                for index, question in enumerate(pending_questions):
                    answers[question] = st.text_input(
                        question, key=f"ask_more_answer::{source}::{index}", disabled=busy
                    )
                if st.button("Submit answers", key=f"ask_more_submit::{source}", disabled=busy):
                    st.session_state[f"feedback_pending::{source}"] = "\n".join(
                        f"Q: {q}\nA: {a}" for q, a in answers.items()
                    )
                    st.session_state[regen_key] = True
                    st.session_state[pending_key] = None
                    st.rerun()
            else:
                feedback = st.text_area(
                    "Feedback — request changes or details to include",
                    key=f"feedback::{result['thread_id']}::{source}",
                    disabled=busy,
                )

                ask_more_col, regen_col, launch_col = st.columns(3)
                if ask_more_col.button("Ask me more", key=f"ask_more::{source}", disabled=busy):
                    st.session_state[ask_more_key] = True
                    st.rerun()

                if regen_col.button(
                    "Regenerate ADR", key=f"regenerate::{source}", disabled=busy or not feedback
                ):
                    st.session_state[f"feedback_pending::{source}"] = feedback
                    st.session_state[regen_key] = True
                    st.rerun()

                if launch_col.button("Publish", key=f"launch::{source}", type="primary", disabled=busy):
                    finalize_result = api_client.finalize_document(tenant, source, candidate["document"])
                    candidates[source]["finalized"] = True
                    candidates[source]["finalized_version"] = finalize_result["version"]
                    st.rerun()

            if st.session_state[ask_more_key]:
                with st.spinner("Drafting new clarification questions — same question limit as the original upload..."):
                    try:
                        # The number the reviewer set on "Input transcription" — "Ask me more"
                        # reuses it rather than asking again, per the user's own request.
                        max_questions = st.session_state.get("upload_max_questions", 10)
                        ask_result = api_client.ask_more_questions(tenant, source, max_questions)
                        if ask_result["questions"]:
                            st.session_state[pending_key] = ask_result["questions"]
                        else:
                            st.session_state[no_questions_key] = True
                    finally:
                        st.session_state[ask_more_key] = False
                st.rerun()

            if st.session_state[regen_key]:
                with st.spinner("Regenerating the ADR with your feedback..."):
                    try:
                        regen_result = api_client.regenerate_document(
                            tenant, source, st.session_state[f"feedback_pending::{source}"]
                        )
                        candidates[source] = {
                            **candidate,
                            "document": regen_result["document"],
                            "score": regen_result["score"],
                            "unresolved_points": regen_result["unresolved_points"],
                        }
                    finally:
                        st.session_state[regen_key] = False
                st.rerun()


def _input_transcription_tab(tenant: str) -> None:
    st.subheader("Input transcription")
    st.caption("Upload transcriptions and PDFs for one meeting, to be processed and clarified.")

    uploading = st.session_state.setdefault("upload_in_progress", False)

    ingestion_date = st.date_input("Meeting date", value=date.today(), disabled=uploading)
    uploaded = st.file_uploader(
        "Transcript (.vtt, .txt, .md) or notes (.pdf) — you can select several files for the same meeting",
        type=["txt", "vtt", "md", "pdf"],
        accept_multiple_files=True,
        disabled=uploading,
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
    )

    if st.button("Process and clarify", disabled=uploading or not uploaded, type="primary"):
        # Snapshot the file bytes and the click's own inputs into session_state, then rerun
        # once *before* the actual (slow, real-LLM) call — this is what lets the button below
        # render as disabled while the call is in flight. Without this extra rerun, the button
        # would only re-render disabled AFTER the blocking call already finished, leaving a
        # window where an impatient extra click starts a second, separately-billed graph run.
        st.session_state.upload_payload = [(f.name, f.getvalue()) for f in uploaded]
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
                    tenant,
                    st.session_state.upload_ingestion_date,
                    st.session_state.upload_payload,
                    st.session_state.upload_max_questions,
                )
            except requests.HTTPError as exc:
                st.error(f"Upload failed: {exc.response.json().get('detail', exc)}")
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
        # Keyed by position, not by question text: the backend de-dupes exact-duplicate
        # questions (agents.graph._top_questions), but keying widgets off arbitrarily long,
        # LLM-generated question text is fragile regardless — an index is always unique and
        # always short.
        for index, question in enumerate(result["pending_questions"]):
            answers[question] = st.text_input(
                question, key=f"answer::{result['thread_id']}::{index}", disabled=resuming
            )

        if st.button("Submit answers", disabled=resuming):
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
                        st.session_state.resume_thread_id, st.session_state.resume_answers
                    )
                finally:
                    st.session_state.resume_in_progress = False
            st.rerun()
    else:
        st.success(f"Done. Files ingested: {', '.join(result['files_ingested']) or '(none — already ingested)'}")
        _render_adr_candidates(tenant, result)


def _architecture_history_tab(tenant: str) -> None:
    st.subheader("Architecture history")
    st.caption(
        "Current architecture, built live from Gold's own component state, and every ADR "
        "published so far."
    )

    history = api_client.architecture_history(tenant)

    if history["diagram"]:
        st.mermaid_chart(history["diagram"])
    else:
        st.info("No components in Gold yet — publish an ADR from Input transcription to see it here.")

    st.write("#### ADRs")
    if not history["adrs"]:
        st.info("No ADRs published yet.")
        return

    # A real `st.dataframe` (not hand-rolled `st.columns` rows) so "View ADR" can be a genuine
    # link cell (`LinkColumn`) that opens in a new browser tab — same query-param scheme
    # `gold_service.current_architecture_diagram`'s own node links use, read back by `main()`'s
    # `_adr_viewer_page` branch. Opening that link starts a brand new Streamlit session (a
    # separate browser tab always does), so it asks for login again — not a defect, just how
    # Streamlit tabs work.
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


def _adr_viewer_page(tenant: str, source_component: str, version: int) -> None:
    """The landing page a "View ADR" link opens in its new tab (`?view_adr=...&
    view_adr_version=...`, read by `main()`) — a single read-only ADR, looked up from the same
    `architecture_history` payload the table itself renders from, no separate endpoint."""
    st.caption(f"Architecture history — {source_component} v{version}")
    history = api_client.architecture_history(tenant)
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


def _chat_tab(tenant: str) -> None:
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
            result = api_client.chat(tenant, question)
        history.append(("assistant", result["answer"]))
        st.rerun()


def _test_monitor_tab() -> None:
    st.subheader("Test monitor")
    st.caption("Test suite results and LLM token/cost consumption for the whole application.")

    try:
        data = api_client.test_monitor()
    except requests.HTTPError:
        st.info("Test monitor is disabled (start_test_mode is off).")
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
        nav_items.append("Test monitor")
    active_page = st.session_state.setdefault("active_page", nav_items[0])
    if active_page not in nav_items:  # e.g. Test monitor got disabled mid-session
        active_page = nav_items[0]

    with st.sidebar:
        st.markdown(f"**{user['username']}**")
        st.caption(f"tenant: {user['tenant']}")
        st.markdown("---")
        # Vertical nav: each item is a full-width button. The CURRENT page renders as
        # type="primary" purely as a hook for theme.py's CSS to style it as "active" (filled,
        # bright text) versus every other item's plain "inactive" look — it never triggers
        # theme.py's own purple primary-button styling, which is scoped to the main content
        # area only, not the sidebar.
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
        _input_transcription_tab(user["tenant"])
    elif active_page == "Architecture history":
        _architecture_history_tab(user["tenant"])
    elif active_page == "Chat with RAG":
        _chat_tab(user["tenant"])
    elif active_page == "Test monitor":
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

    # A "View ADR" link (`_architecture_history_tab`'s table, or a node in
    # `gold_service.current_architecture_diagram`) always opens in a NEW browser tab with these
    # query params — a fresh Streamlit session, hence the login gate above still applies first.
    # Once logged in, this branch takes over the whole page instead of the normal sidebar nav.
    source_component = st.query_params.get("view_adr")
    version = st.query_params.get("view_adr_version")
    if source_component and version:
        _adr_viewer_page(st.session_state.user["tenant"], source_component, int(version))
        return

    _main_app()


if __name__ == "__main__":
    main()
