"""Compact theme for the Streamlit frontend. Injects one CSS string per page load.
Also has small HTML helpers, for example the completeness score bar (`st.progress`
gives no color control).

The sidebar is dark. The main content area is light, for best reading contrast.

The CSS targets stable `data-testid`/`data-baseweb` attributes, not Streamlit's
internal class names. Those names are hashed and change between versions.
"""

import base64
import html
from pathlib import Path

import streamlit as st

# Reads and encodes the logo once per process, at import time, not on every Streamlit rerun.
# Path is relative to the repo root (the working directory in both local and Docker runs).
# Embedded as a base64 data URI: the browser cannot load a server-side file path directly.
_LOGO_PATH = Path("app/img/logo.png")
_LOGO_DATA_URI = (
    f"data:image/png;base64,{base64.b64encode(_LOGO_PATH.read_bytes()).decode('ascii')}"
    if _LOGO_PATH.exists()
    else ""
)

# Height of the fixed header bar. Every other fixed/absolute layout piece shifts down by
# this much, so nothing renders under the header. See the CSS "HEADER BAR" section.
HEADER_HEIGHT_PX = 64

CSS = f"""
/* ================= FULL-WIDTH PAGE HEADER =================
   Plain top bar (logo + title), spanning the full browser width, over the sidebar.
   `position: fixed` escapes the container a normal Streamlit element is stuck inside.
   `z-index` stays above `stHeader` and `stSidebar`. `.stApp` then shifts down by
   `HEADER_HEIGHT_PX`, so the sidebar and main content both start below the bar. */
.app-header-bar {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: {HEADER_HEIGHT_PX}px;
    z-index: 1000000;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 20px;
    box-sizing: border-box;
}}
.app-header-bar img {{
    height: 36px;
    width: 36px;
    border-radius: 8px;
    object-fit: cover;
    flex-shrink: 0;
}}
.app-header-bar .app-header-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}}
/* `stAppViewContainer` holds both the sidebar and the main content, and is
   `position: absolute; top: 0`. Shift it down directly. `.stApp`'s `padding-top` does not
   work here: padding cannot move an absolutely positioned box's own origin. */
[data-testid="stAppViewContainer"] {{
    top: {HEADER_HEIGHT_PX}px !important;
    height: calc(100vh - {HEADER_HEIGHT_PX}px) !important;
}}
"""

CSS += """
:root {
    /* ---- Sidebar (dark) ---- */
    --sidebar-bg: #262931;
    --sidebar-surface: #23252b;
    --sidebar-surface-elevated: #2b2e37;
    --sidebar-surface-hover: #30333d;
    --sidebar-border: #454957;
    --sidebar-border-subtle: #383b45;
    --sidebar-text-primary: #f1f3f5;
    --sidebar-text-secondary: #aeb3bd;
    --sidebar-text-muted: #747b88;

    /* ---- Main content (light) ---- */
    --background: #ffffff;
    --surface: #f8f9fa;
    --surface-elevated: #f1f3f5;
    --surface-hover: #eceef1;
    --border: #dcdfe4;
    --border-subtle: #e9ecef;
    --text-primary: #16181d;
    --text-secondary: #4b505c;
    --text-muted: #868e96;

    /* ---- Accents — one hue set, two uses ----
       *-fill: the bright palette, for solid fills (buttons, badges, score bar).
       *-text: a darker read of the same hue, for text on the white background (links). */
    --purple: #8b5cf6;
    --purple-text: #6d28d9;
    /* Fill for the disabled primary button. Dark enough to keep white label text readable. */
    --purple-muted: #a99bd1;
    --cyan: #22d3ee;
    --cyan-text: #0e7490;
    --green: #34d399;
    --green-text: #047857;
    --yellow: #facc15;
    --yellow-text: #a16207;
    --orange: #fb923c;
    --orange-text: #c2410c;
    --red: #f87171;
    --red-text: #b91c1c;
    --blue: #60a5fa;
    --blue-text: #1d4ed8;

    --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

/* ================= MAIN CONTENT — light ================= */

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--background);
}
.stApp, .stApp p, .stApp span, .stApp label, .stMarkdown {
    color: var(--text-secondary);
}

/* Small titles on purpose — compact, technical look. */
h1 { font-size: 1.35rem !important; }
h2 { font-size: 1.1rem !important; }
h3 { font-size: 0.95rem !important; }
h4 { font-size: 0.85rem !important; }
h1, h2, h3, h4, .stApp strong {
    color: var(--text-primary) !important;
    font-weight: 600;
    letter-spacing: -0.01em;
}
[data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}
[data-testid="stAppViewContainer"] a { color: var(--cyan-text); }

[data-testid="stAppViewContainer"] code,
[data-testid="stAppViewContainer"] pre,
[data-testid="stAppViewContainer"] [data-testid="stCodeBlock"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
}

/* ---- Buttons (main content only — the sidebar overrides these again below) ---- */
[data-testid="stAppViewContainer"] .stButton button,
[data-testid="stAppViewContainer"] .stFormSubmitButton button {
    background: var(--surface-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
}
[data-testid="stAppViewContainer"] .stButton button:hover,
[data-testid="stAppViewContainer"] .stFormSubmitButton button:hover {
    background: var(--surface-hover);
    border-color: var(--text-muted);
    color: var(--text-primary);
}
[data-testid="stAppViewContainer"] .stButton button[kind="primary"],
[data-testid="stAppViewContainer"] .stFormSubmitButton button[kind="primary"] {
    background: var(--purple);
    border-color: var(--purple);
}
/* Targets every descendant, not just the button. Streamlit wraps the label in nested
   <div>/<span>/<p> tags. The generic `.stApp span`/`.stApp p` rule sets color directly
   on those, and a direct rule always wins over an inherited one. */
[data-testid="stAppViewContainer"] .stButton button[kind="primary"],
[data-testid="stAppViewContainer"] .stButton button[kind="primary"] *,
[data-testid="stAppViewContainer"] .stFormSubmitButton button[kind="primary"],
[data-testid="stAppViewContainer"] .stFormSubmitButton button[kind="primary"] * {
    color: #ffffff !important;
}
[data-testid="stAppViewContainer"] .stButton button[kind="primary"]:hover {
    background: var(--purple-text);
    border-color: var(--purple-text);
}
/* Disabled primary buttons. The generic `:disabled` rule below sets a near-white fill,
   which with the forced white text above left the label unreadable. More specific than
   that rule, so it wins without `!important`. */
[data-testid="stAppViewContainer"] .stButton button[kind="primary"]:disabled,
[data-testid="stAppViewContainer"] .stFormSubmitButton button[kind="primary"]:disabled {
    background: var(--purple-muted);
    border-color: var(--purple-muted);
}
[data-testid="stAppViewContainer"] .stButton button:disabled {
    background: var(--surface);
    color: var(--text-muted);
    border-color: var(--border-subtle);
}

/* Per-question action buttons (Irrelevant / Infer an answer / Suggest info).
   `_render_question_row` gives each a `key` starting with `qbtn-<action>-`, which Streamlit
   turns into a `st-key-<key>` class. `[class*=...]` matches that class for any question.
   `!important` makes these colors win over the plain/primary button rules above.
   Color is also set on every descendant, for the same reason as the primary-button fix
   above. Smaller font keeps "Infer an answer" from being clipped. */
[data-testid="stAppViewContainer"] [class*="st-key-qbtn-irrelevant-"] button,
[data-testid="stAppViewContainer"] button[class*="st-key-qbtn-irrelevant-"],
[data-testid="stAppViewContainer"] [class*="st-key-qbtn-irrelevant-"] button *,
[data-testid="stAppViewContainer"] button[class*="st-key-qbtn-irrelevant-"] * {
    background: var(--red) !important;
    border-color: var(--red-text) !important;
    color: #ffffff !important;
}
[data-testid="stAppViewContainer"] [class*="st-key-qbtn-infer-"] button,
[data-testid="stAppViewContainer"] button[class*="st-key-qbtn-infer-"],
[data-testid="stAppViewContainer"] [class*="st-key-qbtn-infer-"] button *,
[data-testid="stAppViewContainer"] button[class*="st-key-qbtn-infer-"] * {
    background: var(--orange) !important;
    border-color: var(--orange-text) !important;
    color: #ffffff !important;
}
[data-testid="stAppViewContainer"] [class*="st-key-qbtn-suggest-"] button,
[data-testid="stAppViewContainer"] button[class*="st-key-qbtn-suggest-"],
[data-testid="stAppViewContainer"] [class*="st-key-qbtn-suggest-"] button *,
[data-testid="stAppViewContainer"] button[class*="st-key-qbtn-suggest-"] * {
    background: var(--green) !important;
    border-color: var(--green-text) !important;
    color: #ffffff !important;
}
[data-testid="stAppViewContainer"] [class*="st-key-qbtn-"] button {
    font-size: 0.68rem !important;
    padding: 6px 4px !important;
    height: auto !important;
    min-height: 2.2rem !important;
    line-height: 1.15 !important;
    white-space: normal !important;
}
/* The label sits in a Streamlit text-wrapper div that defaults to
   `overflow:hidden; text-overflow:ellipsis; white-space:nowrap`. This clipped "Infer an
   answer" to "Infer an…" even after the font-size cut above. Forces wrap instead, so the
   label breaks onto a second line. */
[data-testid="stAppViewContainer"] [class*="st-key-qbtn-"] button * {
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: normal !important;
    word-break: break-word !important;
}

/* ---- Inputs ---- */
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input {
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 1px var(--purple) !important;
}

/* ---- File uploader ---- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 8px;
}
[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-secondary) !important;
}

/* ---- Expander ---- */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p {
    color: var(--text-primary) !important;
}

/* ---- Alerts ---- */
div[data-testid="stAlert"] {
    border-radius: 6px;
    border-width: 1px;
    border-style: solid;
    background: var(--surface);
}

/* ---- Chat ---- */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
}
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

/* ---- Table ---- */
[data-testid="stTable"] table, [data-testid="stDataFrame"] {
    background: var(--surface);
    color: var(--text-secondary);
}

/* ================= SIDEBAR — dark, kept distinct on purpose ================= */

section[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--sidebar-border-subtle);
}
section[data-testid="stSidebar"] * {
    color: var(--sidebar-text-secondary) !important;
}
section[data-testid="stSidebar"] strong, section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: var(--sidebar-text-primary) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--sidebar-border-subtle);
}

/* Vertical nav. Flat, no border. Pill-shaped highlight only on the current page. */
section[data-testid="stSidebar"] .stButton button {
    background: transparent;
    color: var(--sidebar-text-secondary);
    border: none;
    text-align: left;
    justify-content: flex-start;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: var(--sidebar-surface-hover);
    color: var(--sidebar-text-primary);
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: var(--sidebar-surface-elevated);
    color: var(--sidebar-text-primary);
    border: none;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    background: var(--sidebar-surface-elevated);
}
"""


def inject() -> None:
    """Injects the theme's `<style>` block and the full-width header (logo and title).
    Call once, early in `main()`, on every script run — cheap, since Streamlit reruns
    the whole script anyway.

    `main()` calls this before its login check. The header shows on every page,
    including login, with no extra work per page."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="app-header-bar">
            <img src="{_LOGO_DATA_URI}" alt="logo" />
            <span class="app-header-title">Architecture evolution RAG v.0.1</span>
        </div>""",
        unsafe_allow_html=True,
    )


def score_bar_html(score: int) -> str:
    """Builds a small HTML completeness bar. Color depends on the score band: green,
    yellow, or red. `score` is a number from 0 to 100."""
    if score >= 80:
        color = "var(--green)"
    elif score >= 50:
        color = "var(--yellow)"
    else:
        color = "var(--red)"
    return f"""
    <div style="margin: 4px 0 12px 0;">
      <div style="display:flex; justify-content:space-between; font-family:var(--mono);
                  font-size:0.72rem; color:var(--text-muted); margin-bottom:4px;
                  text-transform:uppercase; letter-spacing:0.04em;">
        <span>Completeness</span><span style="color:{color};">{score}%</span>
      </div>
      <div style="background:var(--surface-elevated); border-radius:6px; height:8px;
                  overflow:hidden; border:1px solid var(--border-subtle);">
        <div style="width:{score}%; background:{color}; height:100%;"></div>
      </div>
    </div>
    """


def adr_metadata_header_html(
    source_component: str,
    version: int,
    ingestion_date: str,
    authored_by: str,
    created_at: str,
    content_hash: str,
) -> str:
    """Builds the metadata header shown above a published ADR, on the "View ADR" page.
    Blue, not yellow: yellow is reserved for the Gold-entity cards below, so the two
    blocks look distinct."""
    fields = [
        ("Meeting date", ingestion_date),
        ("Authored by", authored_by or "—"),
        ("Generated at", created_at.split("T", 1)[0]),
        ("Content hash", content_hash[:12]),
    ]
    fields_html = "".join(
        f"""<div>
              <div style="font-family:var(--mono); font-size:0.68rem; color:var(--blue-text);
                          text-transform:uppercase; letter-spacing:0.04em; margin-bottom:2px;">{html.escape(label)}</div>
              <div style="font-size:0.85rem; color:var(--text-primary);">{html.escape(str(value))}</div>
            </div>"""
        for label, value in fields
    )
    return f"""
    <div style="background:rgba(96, 165, 250, 0.08); border:1px solid rgba(96, 165, 250, 0.35);
                border-left:4px solid var(--blue); border-radius:8px; padding:14px 18px; margin:4px 0 16px 0;">
      <div style="font-size:1.05rem; font-weight:700; color:var(--text-primary); margin-bottom:10px;">
        {html.escape(source_component)} <span style="color:var(--blue-text); font-weight:600;">v{version}</span>
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:20px;">
        {fields_html}
      </div>
    </div>
    """


# Maps a `gold_evolution.operation` value to a color. Covers `ComponentStatus`,
# `ContractAction`, and `ArchitectureChangeType` in one dict, since a card does not know
# its entity_type in advance. Unknown values fall back to `--text-muted`.
_OPERATION_COLORS = {
    "new": "var(--green-text)",
    "forward-update": "var(--blue-text)",
    "modified": "var(--blue-text)",
    "changed": "var(--blue-text)",
    "break-change": "var(--red-text)",
    "removed": "var(--red-text)",
    "deprecated": "var(--orange-text)",
    "unchanged": "var(--text-muted)",
    "unknown": "var(--text-muted)",
}


def gold_entity_cards_html(entities: list[dict]) -> str:
    """Builds the "Generated in Gold" card row at the end of a published ADR. One card per
    `gold_evolution` row this ADR version wrote. Each `entities` item needs `entity_type`,
    `canonical_name`, `operation`, and `version` keys.

    Soft yellow background sets this block apart, so a reader sees at a glance what this
    ADR added to Gold."""
    if not entities:
        return (
            '<p style="color:var(--text-muted); font-size:0.85rem;">'
            "This ADR version did not add or change anything in Gold.</p>"
        )
    cards = "".join(
        f"""<div style="background:rgba(250, 204, 21, 0.10); border:1px solid rgba(250, 204, 21, 0.4);
                    border-radius:8px; padding:10px 14px; min-width:200px;">
              <div style="font-family:var(--mono); font-size:0.65rem; color:var(--yellow-text);
                          text-transform:uppercase; letter-spacing:0.04em;">{html.escape(entity["entity_type"])}</div>
              <div style="font-size:0.9rem; font-weight:600; color:var(--text-primary); margin:2px 0 4px 0;">
                {html.escape(entity["canonical_name"])}
              </div>
              <div style="font-size:0.75rem; color:{_OPERATION_COLORS.get(entity["operation"], "var(--text-muted)")};">
                {html.escape(entity["operation"])} · v{entity["version"]}
              </div>
            </div>"""
        for entity in entities
    )
    return f"""
    <div style="margin-top:6px;">
      <div style="font-family:var(--mono); font-size:0.72rem; color:var(--text-muted);
                  text-transform:uppercase; letter-spacing:0.04em; margin-bottom:8px;">
        Generated in Gold — {len(entities)} entit{"y" if len(entities) == 1 else "ies"}
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:10px;">
        {cards}
      </div>
    </div>
    """
