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

import streamlit as st

CSS = """
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
    color: #ffffff;
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
    """Injects the theme's `<style>` block — call this once, early in `main()`, on every
    script run (cheap; Streamlit re-runs the whole script on every interaction anyway)."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


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
