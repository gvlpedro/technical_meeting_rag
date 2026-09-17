"""Compact, developer-oriented theme for the Streamlit frontend — one CSS string injected
once per page load, plus a couple of small HTML-rendering helpers for the one element (the
completeness score bar) worth more control than a native Streamlit widget gives.

Two zones, deliberately different: the left sidebar (the vertical nav + who's logged in)
stays dark, so it reads as a distinct rail; the main content area is light (white background,
near-black text) for maximum reading contrast on the actual work — uploaded transcripts,
generated ADRs, chat answers. Targets Streamlit's own stable `data-testid`/`data-baseweb`
attributes rather than its internal (hashed, version-fragile) class names — the same
technique Streamlit's own docs recommend for custom CSS.
"""

import base64
from pathlib import Path

import streamlit as st

# Read once per process (module-level, not per rerun) — Streamlit reruns this whole script on
# every interaction, but re-importing an already-imported module is a no-op, so this only ever
# reads+encodes the file once. Relative to the CWD (repo root, both locally via
# `uv run streamlit run frontend/app.py` and in Docker via the image's WORKDIR) — the same
# convention the rest of the app already uses for on-disk paths. Base64-embedded as a data URI
# because a plain `<img src="app/img/logo.png">` inside `unsafe_allow_html` HTML can't resolve a
# server-side filesystem path — the browser has no access to it.
_LOGO_PATH = Path("app/img/logo.png")
_LOGO_DATA_URI = (
    f"data:image/png;base64,{base64.b64encode(_LOGO_PATH.read_bytes()).decode('ascii')}"
    if _LOGO_PATH.exists()
    else ""
)

# Height of the fixed full-width header bar `render_header` draws — every other fixed/absolute
# Streamlit layout piece (`stHeader`, `stSidebar`, `stAppViewContainer`) gets pushed down by
# exactly this much (see the CSS's "HEADER BAR" section) so nothing renders underneath it.
HEADER_HEIGHT_PX = 64

CSS = f"""
/* ================= FULL-WIDTH PAGE HEADER =================
   A plain top bar — logo + bold title, no menu — spanning the ENTIRE browser width, including
   over the sidebar, like a website's top nav bar. `position: fixed` escapes whatever column/
   container `render_header`'s `st.markdown` call happens to sit inside (that's what makes a
   normal Streamlit element "full width" in the first place: it can only ever fill its own
   container, never the viewport) and a `z-index` above both `stHeader` (999990) and `stSidebar`
   (999991, confirmed on the live DOM) keeps it on top of both. Every other top-level layout
   piece is then shifted down by `HEADER_HEIGHT_PX` so it starts below the bar instead of
   underneath it — `.stApp` is the single outermost container both the sidebar and the header/
   main content live inside, so shifting it once pushes everything uniformly. */
.app-header-bar {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: {HEADER_HEIGHT_PX}px;
    z-index: 1000000;
    background: var(--background);
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
/* `[data-testid="stAppViewContainer"]` is the one container holding BOTH the sidebar and the
   main content — itself `position: absolute; top: 0`, so shifting IT down (rather than
   `.stApp`'s `padding-top`, which does nothing here: padding never moves an absolutely
   positioned box's own origin, confirmed on the live DOM — the sidebar and its collapse button
   stayed at `top: 0`, hidden under this bar, even with that padding in place) carries the
   sidebar, its collapse button, `stHeader`'s Deploy/menu, and the main content down together as
   one unit, preserving whatever positioning each already had relative to it. */
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
       *-fill: exactly the given palette, for solid fills (buttons, badges, the score bar) —
       a light/dark foreground sits ON TOP of these, so the original bright values are fine.
       *-text: a darker read of the same hue, for the accent used AS text color on the white
       main background (links) — the bright fill values are too light to pass contrast there. */
    --purple: #8b5cf6;
    --purple-text: #6d28d9;
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

/* Relatively small titles, on purpose — compact/technical, not a marketing page. */
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
/* `!important` + targeting every descendant, not just the button itself: Streamlit wraps a
   button's label in nested <div>/<span>/<p> elements, and the generic `.stApp span`/`.stApp p`
   rule above sets its own `color` directly on those — a direct rule on an element always wins
   over an inherited one, regardless of the ancestor selector's specificity. Without this, the
   label read as --text-secondary gray on the purple fill even though the <button> itself
   computed white correctly (confirmed via the actual rendered DOM, not just the stylesheet). */
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
[data-testid="stAppViewContainer"] .stButton button:disabled {
    background: var(--surface);
    color: var(--text-muted);
    border-color: var(--border-subtle);
}

/* Per-question clarification action buttons (Irrelevant / Infer an answer / Suggest info) —
   `frontend/app.py::_render_question_row` gives each one a `key` starting with `qbtn-<action>-`,
   which Streamlit turns into a `st-key-<key>` class it attaches near the button; `[class*=...]`
   matches that class regardless of the unique per-question suffix or whether Streamlit puts it
   on the button itself or a wrapping element. `!important` so these permanent colors always win
   over the plain-button/primary-button rules above, regardless of rule order.

   Text color is set on the button AND on every descendant (`* { color: ... !important }`) for
   the same reason the primary-button fix above needs it: Streamlit wraps the label in nested
   <div>/<span>/<p> elements that `.stApp span`/`.stApp p` give their own direct `color`, which
   otherwise wins over whatever the <button> itself computes (confirmed against the live DOM).
   White text (not the previous dark-on-bright shades) is what actually reads on these
   saturated fills; the smaller font keeps "Infer an answer" from being clipped in a ~13%-wide
   button (see `_render_question_row`'s 60/40 answer/buttons split). */
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
/* The label sits inside a Streamlit-generic text-wrapper div (not one of our own classes) that
   defaults to `overflow:hidden; text-overflow:ellipsis; white-space:nowrap` — fine for the
   arbitrary-length labels it's normally built for, but it was clipping "Infer an answer" to
   "Infer an…" even after the font-size cut above (confirmed on the live DOM: that div's own
   `clientWidth` was 58px against a 90px `scrollWidth`). Forcing wrap on every descendant here,
   scoped to just these three buttons, lets the label break onto a second line instead of being
   cut — the buttons are short enough that two lines still reads fine. */
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

/* Vertical nav — plain "inactive" items and one "active" (primary) item, matching the exact
   top-nav look this app used before the layout moved into the sidebar: flat, no border,
   pill-shaped highlight only on the current page. */
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
    """Injects the theme's `<style>` block AND the full-width page header (logo + bold title,
    see the CSS's "FULL-WIDTH PAGE HEADER" section) — call this once, early in `main()`, on
    every script run (cheap; Streamlit re-runs the whole script on every interaction anyway).
    Called unconditionally before `main()`'s login check, so the header is the very first thing
    on every page — the login screen included — not something each page has to remember to
    render for itself."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="app-header-bar">
            <img src="{_LOGO_DATA_URI}" alt="logo" />
            <span class="app-header-title">Technical Meeting RAG</span>
        </div>""",
        unsafe_allow_html=True,
    )


def score_bar_html(score: int) -> str:
    """A small HTML completeness bar, colored by band (green/yellow/red) — more control over
    the "is this actually production-ready" signal than `st.progress`'s single fixed color
    gives. `score` is 0-100."""
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
