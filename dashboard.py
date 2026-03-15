from pathlib import Path
import pickle
import warnings
import boto3

import pandas as pd
import plotly.express as px
import streamlit as st


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH       = Path("cleaned_reviews.csv")
MODEL_PATH      = Path("sentiment_model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")

REQUIRED_COLUMNS       = {"Clothing ID", "Rating", "sentiment", "intent"}
TEXT_COLUMN_CANDIDATES = ["Review Text", "clean_review", "review"]

SENTIMENT_COLORS = {"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#F39C12"}
INTENT_COLORS    = {
    "other": "#95A5A6", "size issue": "#3498DB", "quality issue": "#E74C3C",
    "design feedback": "#9B59B6", "price issue": "#F39C12",
}
DEMO_REVIEWS = [
    "Absolutely wonderful — silky and comfortable, fits perfectly, would buy again",
    "The fabric is cheap and the sizing is way off. Very disappointed.",
    "Nice design but overpriced for the quality, stitching came apart after one wash.",
]
AT_RISK_THRESHOLD = 6.0
NOVA_MODEL_ID     = "amazon.nova-pro-v1:0"

# ── Google Fonts import ───────────────────────────────────────────────────────
FONTS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');
"""

# ── Global CSS ────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080809 !important;
    color: #e8e6e0 !important;
    font-size: 13px !important;
}
/* Prevent Syne from inheriting huge sizes */
[data-testid="stMainBlockContainer"] * {
    font-family: 'Syne', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 110% -10%, rgba(238,121,39,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at -10% 110%, rgba(91,140,245,0.08) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 47px, rgba(255,255,255,0.025) 48px),
        repeating-linear-gradient(90deg, transparent, transparent 47px, rgba(255,255,255,0.025) 48px),
        #080809 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d0f !important;
    border-right: 1px solid rgba(238,121,39,0.15) !important;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }
[data-testid="stSidebar"] h2 {
    font-family: 'DM Serif Display', serif !important;
    color: #ee7927 !important;
    font-size: 22px !important;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] label {
    color: rgba(232,230,224,0.75) !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #ee7927 !important;
}
[data-testid="stSidebar"] .stSlider div[role="slider"] {
    background: #ee7927 !important;
    box-shadow: 0 0 8px rgba(238,121,39,0.6) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: rgba(232,230,224,0.45) !important;
    border-radius: 9px !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #ee7927, #c85e14) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(238,121,39,0.35) !important;
}
[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {
    color: #e8e6e0 !important;
    background: rgba(255,255,255,0.06) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-top: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 14px !important;
    padding: 16px 14px !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    min-width: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
    height: 90px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(238,121,39,0.4) !important;
    box-shadow: 0 0 20px rgba(238,121,39,0.12) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(232,230,224,0.75) !important;
    margin-bottom: 4px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.5rem !important;
    color: #ffffff !important;
    line-height: 1.1 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.5rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    margin-top: 2px !important;
}

/* ── Plotly chart cards ── */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 20px !important;
    padding: 18px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow:
        0 24px 48px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.08) !important;
    margin-bottom: 20px !important;
    overflow: hidden !important;
    transition: box-shadow 0.3s, border-color 0.3s !important;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow:
        0 32px 64px rgba(0,0,0,0.6),
        0 0 32px rgba(238,121,39,0.06),
        inset 0 1px 0 rgba(255,255,255,0.1) !important;
    border-color: rgba(238,121,39,0.18) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e6e0 !important;
    font-family: 'Syne', sans-serif !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #ee7927 !important;
    box-shadow: 0 0 0 3px rgba(238,121,39,0.1) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.05) !important;
    color: rgba(232,230,224,0.8) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    background: rgba(238,121,39,0.15) !important;
    border-color: rgba(238,121,39,0.4) !important;
    color: #ee7927 !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #ee7927, #c85e14) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(238,121,39,0.35) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(238,121,39,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e6e0 !important;
}

/* ── Multiselect tags ── */
span[data-baseweb="tag"] {
    background: linear-gradient(135deg, #ee7927, #c85e14) !important;
    color: #fff !important;
    border-radius: 6px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(238,121,39,0.25) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Syne', sans-serif !important;
    color: #e8e6e0 !important;
    font-size: 13px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

/* ── Section headers ── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #e8e6e0 !important;
}
h2 { font-size: 18px !important; letter-spacing: -0.01em; }
h3 { font-size: 15px !important; }

/* ── Captions / small text ── */
[data-testid="stCaptionContainer"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    color: rgba(232,230,224,0.45) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 3px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(238,121,39,0.3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(238,121,39,0.6); }

/* ── Vertical divider between columns ── */
.right-panel {
    border-left: 1px solid rgba(238,121,39,0.12);
    padding-left: 24px !important;
}


/* ── Chat column top-aligned ── */
div[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}
div[data-testid="stHorizontalBlock"] > div {
    align-self: flex-start !important;
}

/* ── Fix general font size to 13px everywhere ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    font-size: 13px !important;
    font-family: 'Syne', sans-serif !important;
    color: rgba(232,230,224,0.85) !important;
}

/* ── Fix tab content font ── */
[data-testid="stTabsContent"] {
    font-size: 13px !important;
}

/* ── Ensure stMetric doesnt truncate ── */
[data-testid="stMetric"] > div {
    overflow: visible !important;
    width: 100% !important;
}
[data-testid="stMetricValue"] > div {
    overflow: visible !important;
    white-space: nowrap !important;
    font-size: 1.35rem !important;
}

/* ── Section header text ── */
.section-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 22px !important;
    color: #e8e6e0 !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: rgba(238,121,39,0.08) !important;
    border: 1px solid rgba(238,121,39,0.25) !important;
    color: #ee7927 !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: rgba(238,121,39,0.18) !important;
}
</style>
"""


# ── Data / model loaders ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model_artifacts(mp, vp):
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = Warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pickle.load(open(mp,"rb"))
        v = pickle.load(open(vp,"rb"))
    return m, v


# ── Utilities ─────────────────────────────────────────────────────────────────
def get_text_col(data):
    for c in TEXT_COLUMN_CANDIDATES:
        if c in data.columns: return c
    return None

def validate_data(data):
    return sorted(REQUIRED_COLUMNS - set(data.columns))

def rating_to_score(s):
    return float((s * 2).mean())

def apply_filters(data, rating_range, sel_sent, sel_intent, sel_depts, age_range):
    f = data[data["Rating"].between(*rating_range)]
    if sel_sent:   f = f[f["sentiment"].isin(sel_sent)]
    if sel_intent: f = f[f["intent"].isin(sel_intent)]
    if sel_depts and "Department Name" in f.columns:
        f = f[f["Department Name"].isin(sel_depts)]
    if age_range and "Age" in f.columns:
        f = f[f["Age"].between(*age_range)]
    return f


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _layout(h=300):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8c5bc", family="IBM Plex Mono, monospace", size=11),
        autosize=True, height=h,
        margin=dict(t=44, b=36, l=36, r=20),
        title_font=dict(family="DM Serif Display, serif", size=16, color="#e8e6e0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                   tickfont=dict(family="IBM Plex Mono, monospace", size=10)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                   tickfont=dict(family="IBM Plex Mono, monospace", size=10)),
    )

def dist_bar(series, title, cmap=None):
    df = (series.value_counts(normalize=True).mul(100).round(2)
          .rename_axis("category").reset_index(name="percent"))
    kw = dict(x="category", y="percent", title=title,
              labels={"category":"","percent":"%"}, text_auto=".1f")
    if cmap: kw["color"]="category"; kw["color_discrete_map"]=cmap
    fig = px.bar(df, **kw)
    fig.update_traces(marker_line_width=0, textfont_size=10,
                      textfont_family="IBM Plex Mono, monospace")
    fig.update_layout(showlegend=False, **_layout())
    return fig

def rating_hist(series, title):
    fig = px.histogram(series.to_frame("Rating"), x="Rating", nbins=5, title=title,
                       color_discrete_sequence=["#ee7927"])
    fig.update_traces(marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1)
    fig.update_xaxes(dtick=1)
    fig.update_layout(bargap=0.2, **_layout())
    return fig

def age_hist(series):
    fig = px.histogram(series.to_frame("Age"), x="Age", nbins=20,
                       title="Reviewer Age Distribution",
                       color_discrete_sequence=["#9B59B6"])
    fig.update_traces(marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1)
    fig.update_layout(**_layout())
    return fig

def proba_chart(classes, probs):
    df = pd.DataFrame({"class":classes,"prob":[round(p*100,2) for p in probs]})
    cmap = {c: SENTIMENT_COLORS.get(c,"#95A5A6") for c in classes}
    fig = px.bar(df, x="class", y="prob", color="class", color_discrete_map=cmap,
                 title="Confidence by class (%)", text_auto=".1f")
    fig.update_traces(marker_line_width=0)
    fig.update_layout(showlegend=False, **_layout())
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(data):
    st.sidebar.markdown(
        "<div style='font-family:DM Serif Display,serif;font-size:22px;"
        "color:#ee7927;margin-bottom:4px;padding:8px 0 4px;'>⚡ Filters</div>",
        unsafe_allow_html=True)

    FILTER_KEYS = ["fr","fs","fi","fd","fa"]
    if st.sidebar.button("↺  Reset all", use_container_width=True):
        for k in FILTER_KEYS: st.session_state.pop(k, None)
        st.rerun()

    st.sidebar.markdown("---")

    rmin, rmax = int(data["Rating"].min()), int(data["Rating"].max())
    rr = st.sidebar.slider("Rating", rmin, rmax, (rmin,rmax), key="fr")

    so = sorted(data["sentiment"].dropna().unique().tolist())
    ss = st.sidebar.multiselect("Sentiment", so, so, key="fs")
    eff_s = ss or so

    io = sorted(data["intent"].dropna().unique().tolist())
    si = st.sidebar.multiselect("Intent", io, io, key="fi")
    eff_i = si or io

    sel_d = []
    if "Department Name" in data.columns:
        do = sorted(data["Department Name"].dropna().unique().tolist())
        sel_d = st.sidebar.multiselect("Department", do, [], key="fd")

    age_r = None
    if "Age" in data.columns:
        am, ax = int(data["Age"].min()), int(data["Age"].max())
        age_r = st.sidebar.slider("Age", am, ax, (am,ax), key="fa")

    top_n = st.sidebar.slider("Top N products", 5, 25, 10)

    # Active filter badge
    active = []
    if rr != (rmin,rmax): active.append(f"⭐ {rr[0]}–{rr[1]}")
    if len(eff_s) < len(so): active.append(f"💬 {len(eff_s)} sentiments")
    if len(eff_i) < len(io): active.append(f"🎯 {len(eff_i)} intents")
    if sel_d: active.append(f"🏬 {len(sel_d)} depts")

    if active:
        st.sidebar.markdown(
            f"<div style='font-size:10px;font-family:IBM Plex Mono,monospace;"
            f"color:#ee7927;opacity:0.8;margin-top:8px;line-height:1.8;'>"
            f"{'  ·  '.join(active)}</div>", unsafe_allow_html=True)

    return dict(rr=rr, eff_s=eff_s, eff_i=eff_i, sel_d=sel_d, age_r=age_r, top_n=top_n)


# ── Section header helper ─────────────────────────────────────────────────────
def section_header(title, sub=""):
    st.markdown(
        f"<div style='margin:8px 0 20px;'>"
        f"<div style='font-family:DM Serif Display,serif;font-size:22px;"
        f"color:#e8e6e0;line-height:1.1;'>{title}</div>"
        + (f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;"
           f"letter-spacing:0.1em;color:rgba(232,230,224,0.35);margin-top:4px;'>{sub}</div>"
           if sub else "")
        + "</div>", unsafe_allow_html=True)


# ── Prediction badge ──────────────────────────────────────────────────────────
def prediction_badge(pred, confidence=None):
    color = SENTIMENT_COLORS.get(pred.lower(), "#95A5A6")
    conf_str = f" · {confidence:.1f}% confidence" if confidence else ""
    st.markdown(
        f"<div style='display:inline-flex;align-items:center;gap:10px;"
        f"padding:12px 20px;border-radius:12px;"
        f"background:linear-gradient(135deg,{color}18,{color}08);"
        f"border:1px solid {color}44;margin:12px 0;'>"
        f"<div style='width:10px;height:10px;border-radius:50%;"
        f"background:{color};box-shadow:0 0 8px {color};'></div>"
        f"<span style='font-family:DM Serif Display,serif;font-size:20px;"
        f"color:{color};'>{pred.upper()}</span>"
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:10px;"
        f"color:rgba(232,230,224,0.4);'>{conf_str}</span>"
        f"</div>", unsafe_allow_html=True)



# ── Custom metric card HTML helper ───────────────────────────────────────────
def _metric_card(label, value, delta="", positive=True):
    delta_color = "#2ECC71" if positive else "#E74C3C"
    delta_html  = (f"<div style=\'font-family:IBM Plex Mono,monospace;font-size:10px;"
                   f"color:{delta_color};margin-top:4px;\'>{delta}</div>") if delta else ""
    return f"""<div style=\'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
border-top:1px solid rgba(255,255,255,0.18);border-radius:14px;padding:16px 14px;
height:88px;display:flex;flex-direction:column;justify-content:center;
transition:border-color 0.2s;box-sizing:border-box;\'>
    <div style=\'font-family:IBM Plex Mono,monospace;font-size:10px;letter-spacing:0.1em;
    text-transform:uppercase;color:rgba(232,230,224,0.7);margin-bottom:4px;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;\'>{label}</div>
    <div style=\'font-family:DM Serif Display,serif;font-size:1.45rem;color:#fff;
    line-height:1.1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;\'>{value}</div>
    {delta_html}
</div>"""


# ── Tab: Overview ─────────────────────────────────────────────────────────────
def tab_overview(data, gdata, top_n):
    section_header("Overview", "CUSTOMER MOOD & INTENT ACROSS ALL REVIEWS")
    if data.empty:
        st.info("No data for current filters.")
        return

    g_avg, g_sc = gdata["Rating"].mean(), rating_to_score(gdata["Rating"])

    # Metric row — custom HTML cards for guaranteed equal sizing
    p_score = rating_to_score(data["Rating"])
    d_avg   = data["Rating"].mean() - g_avg
    d_sc    = p_score - g_sc
    rec_html = ""
    if "Recommended IND" in data.columns:
        g_r  = gdata["Recommended IND"].mean()*100
        rr   = data["Recommended IND"].mean()*100
        d_rr = rr - g_r
        rec_html = _metric_card("Rec Rate", f"{rr:.1f}%", f"{d_rr:+.1f}%", d_rr >= 0)
    else:
        rec_html = _metric_card("Rec Rate", "N/A", "", True)

    cards_html = f"""
    <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:24px;'>
        {_metric_card("Reviews",       f"{len(data):,}",           "",                         True)}
        {_metric_card("Avg Rating",    f"{data['Rating'].mean():.2f} ★", f"{d_avg:+.2f}",    d_avg >= 0)}
        {_metric_card("Product Score", f"{p_score:.1f} / 10",      f"{d_sc:+.2f}",            d_sc >= 0)}
        {_metric_card("Products",      f"{data['Clothing ID'].nunique():,}", "",               True)}
        {rec_html}
    </div>"""
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    a,b = st.columns(2)
    with a: st.plotly_chart(dist_bar(data["sentiment"],"Sentiment Distribution",SENTIMENT_COLORS), use_container_width=True)
    with b: st.plotly_chart(dist_bar(data["intent"],"Intent Distribution",INTENT_COLORS), use_container_width=True)

    a2,b2 = st.columns(2)
    with a2: st.plotly_chart(rating_hist(data["Rating"],"Star Rating Distribution"), use_container_width=True)
    with b2:
        if "Age" in data.columns:
            st.plotly_chart(age_hist(data["Age"]), use_container_width=True)

    ps = (data.groupby("Clothing ID")
          .agg(n=("Clothing ID","size"), avg=("Rating","mean"))
          .sort_values("n", ascending=False).head(top_n).reset_index())
    ps["Clothing ID"] = ps["Clothing ID"].astype(str)
    fig = px.bar(ps, x="Clothing ID", y="n", color="avg",
                 color_continuous_scale=[[0,"#E74C3C"],[0.5,"#F39C12"],[1,"#2ECC71"]],
                 color_continuous_midpoint=3.0,
                 title=f"Top {top_n} Products — Volume × Avg Rating",
                 text_auto=True, labels={"n":"Reviews","avg":"Avg ★"})
    fig.update_traces(marker_line_width=0, textfont_family="IBM Plex Mono, monospace")
    fig.update_layout(**_layout(h=340))
    st.plotly_chart(fig, use_container_width=True)


# ── Tab: Product Insights ─────────────────────────────────────────────────────
def tab_product(data, gdata, text_col):
    section_header("Product Insights", "DEEP-DIVE INTO A SINGLE PRODUCT")
    if data.empty:
        st.info("No products for current filters.")
        return

    pid  = st.selectbox("Select Clothing ID", sorted(data["Clothing ID"].dropna().unique()))
    pdat = data[data["Clothing ID"] == pid]
    g_avg, g_sc = gdata["Rating"].mean(), rating_to_score(gdata["Rating"])
    p_avg, p_sc = pdat["Rating"].mean(), rating_to_score(pdat["Rating"])

    # Health banner
    if p_sc < AT_RISK_THRESHOLD:
        st.markdown(
            f"<div style='padding:12px 18px;border-radius:12px;"
            f"background:rgba(231,76,60,0.1);border:1px solid rgba(231,76,60,0.3);"
            f"font-family:Syne,sans-serif;font-size:13px;color:#E74C3C;margin-bottom:16px;'>"
            f"⚠️  At-risk product — score <b>{p_sc:.2f}/10</b> is below {AT_RISK_THRESHOLD}/10 threshold</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='padding:12px 18px;border-radius:12px;"
            f"background:rgba(46,204,113,0.08);border:1px solid rgba(46,204,113,0.25);"
            f"font-family:Syne,sans-serif;font-size:13px;color:#2ECC71;margin-bottom:16px;'>"
            f"✓  Healthy product — score <b>{p_sc:.2f}/10</b></div>",
            unsafe_allow_html=True)

    last_card = (
        _metric_card("Helpful Votes", f"{pdat['Positive Feedback Count'].mean():.2f}", "", True)
        if "Positive Feedback Count" in pdat.columns
        else _metric_card("Unique Intents", str(pdat["intent"].nunique()), "", True)
    )
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;'>
        {_metric_card("Reviews",    f"{len(pdat):,}",         "",              True)}
        {_metric_card("Avg Rating", f"{p_avg:.2f} ★",        f"{p_avg-g_avg:+.2f}", p_avg>=g_avg)}
        {_metric_card("Score",      f"{p_sc:.1f} / 10",      f"{p_sc-g_sc:+.2f}",   p_sc>=g_sc)}
        {last_card}
    </div>""", unsafe_allow_html=True)

    a,b = st.columns(2)
    with a: st.plotly_chart(dist_bar(pdat["sentiment"],"Sentiment Split",SENTIMENT_COLORS), use_container_width=True)
    with b: st.plotly_chart(dist_bar(pdat["intent"],"Intent Split",INTENT_COLORS), use_container_width=True)
    st.plotly_chart(rating_hist(pdat["Rating"],f"Ratings — Product {pid}"), use_container_width=True)

    if text_col:
        st.markdown("<div style='font-family:DM Serif Display,serif;font-size:18px;"
                    "color:#e8e6e0;margin:16px 0 10px;'>Sample Reviews</div>",
                    unsafe_allow_html=True)
        cols = [text_col,"Rating","sentiment","intent"]
        if "Age" in pdat.columns: cols.append("Age")
        st.dataframe(pdat[cols].head(15), use_container_width=True, hide_index=True)


# ── Tab: Review Explorer ──────────────────────────────────────────────────────
def tab_explorer(data, text_col):
    section_header("Review Explorer", "SEARCH & INSPECT RAW REVIEW EVIDENCE")
    if data.empty:
        st.info("No reviews for current filters.")
        return

    q    = st.text_input("🔍  Search reviews", placeholder="quality, fit, size, return...")
    expl = data.copy()
    if q and text_col:
        expl = expl[expl[text_col].fillna("").str.lower().str.contains(q.lower(), na=False)]

    cols = ([text_col] if text_col else []) + ["Clothing ID","Rating","sentiment","intent"]
    if "Age" in data.columns:            cols.append("Age")
    if "Department Name" in data.columns: cols.append("Department Name")

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;'>
        {_metric_card("Matched", f"{len(expl):,}",          "", True)}
        {_metric_card("Shown",   f"{min(len(expl),500):,}", "", True)}
        {_metric_card("Total",   f"{len(data):,}",          "", True)}
    </div>""", unsafe_allow_html=True)

    df_show = expl[cols].head(500)
    st.dataframe(df_show, use_container_width=True, hide_index=True)
    st.caption("Showing up to 500 rows.")
    st.download_button("↓  Download CSV", df_show.to_csv(index=False).encode(),
                       "reviews.csv","text/csv", use_container_width=False)


# ── Tab: Model Insights ───────────────────────────────────────────────────────
def tab_model(text_col, data):
    section_header("Model Insights", "LIVE PREDICTIONS FROM YOUR TRAINED MODEL")
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
        st.warning("Model files not found in project root.")
        return
    try:
        model, vec = load_model_artifacts(MODEL_PATH, VECTORIZER_PATH)
    except Exception as e:
        st.error(f"Load error: {e}")
        return

    # Demo buttons
    st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:10px;"
                "letter-spacing:0.1em;color:rgba(232,230,224,0.4);margin-bottom:8px;'>"
                "QUICK EXAMPLES</div>", unsafe_allow_html=True)
    dcols = st.columns(len(DEMO_REVIEWS))
    for i,(col,txt) in enumerate(zip(dcols, DEMO_REVIEWS)):
        if col.button(f"Example {i+1}", key=f"d{i}", use_container_width=True):
            st.session_state["mta"] = txt

    if "mta" not in st.session_state:
        st.session_state["mta"] = (str(data[text_col].dropna().iloc[0])
                                    if text_col and not data.empty else "")

    user_text = st.text_area("Enter a customer review to predict", height=120, key="mta")

    if st.button("⚡  Predict Sentiment", type="primary", use_container_width=False):
        t = user_text.strip()
        if t:
            X = vec.transform([t])
            pred = str(model.predict(X)[0])
            conf = None
            st.session_state["pred"] = pred
            if hasattr(model,"predict_proba"):
                p = model.predict_proba(X)[0]
                conf = float(max(p)) * 100
                st.session_state["probas"] = list(zip(model.classes_.tolist(), p.tolist()))
                st.session_state["conf"] = conf

    if "pred" in st.session_state:
        prediction_badge(st.session_state["pred"], st.session_state.get("conf"))
        if st.session_state.get("probas"):
            cl, pv = zip(*st.session_state["probas"])
            st.plotly_chart(proba_chart(list(cl), list(pv)), use_container_width=True)

    with st.expander("About this model"):
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;line-height:1.7;"
            "color:rgba(232,230,224,0.7);'>"
            "• Bag-of-words model — word order and context are not captured.<br>"
            "• Sarcasm and mixed sentiment reduce reliability.<br>"
            "• Treat as a decision-support signal, not ground truth.</div>",
            unsafe_allow_html=True)


# ── Bedrock helper ────────────────────────────────────────────────────────────
def bedrock_reply(question, data, history):
    tc = get_text_col(data)
    samples = ""
    if tc:
        s = data[tc].dropna().sample(min(15,len(data)), random_state=42).tolist()
        samples = "\n".join(f"- {str(r)[:200]}" for r in s)

    top5 = (data.groupby("Clothing ID").agg(avg=("Rating","mean"),n=("Rating","size"))
            .sort_values("avg").head(5).reset_index().to_dict(orient="records"))
    bot5 = (data.groupby("Clothing ID").agg(avg=("Rating","mean"),n=("Rating","size"))
            .sort_values("avg",ascending=False).head(5).reset_index().to_dict(orient="records"))

    system = f"""You are RetailIQ Assistant, an expert retail analytics AI.
Dataset: {len(data):,} reviews | avg {data['Rating'].mean():.2f}/5
Sentiment: {data['sentiment'].value_counts().to_dict()}
Intent: {data['intent'].value_counts().to_dict()}
Lowest rated: {top5}
Highest rated: {bot5}
Sample reviews:\n{samples}
Be concise. Bullet points. Under 300 words unless detail is explicitly requested."""

    msgs = [{"role":m["role"],"content":[{"text":m["content"]}]} for m in history[-6:]]
    msgs.append({"role":"user","content":[{"text":question}]})
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    r = client.converse(modelId=NOVA_MODEL_ID, system=[{"text":system}], messages=msgs,
                        inferenceConfig={"maxTokens":600,"temperature":0.5})
    return r["output"]["message"]["content"][0]["text"]


# ── Right AI chat panel ───────────────────────────────────────────────────────
def render_chat_panel(data):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Panel header
    st.markdown("""
        <div style='border-left:2px solid rgba(238,121,39,0.3);padding-left:0;margin-bottom:0;'></div>
        <div style='background:linear-gradient(135deg,#ee7927 0%,#b34d0e 100%);
                    border-radius:16px;padding:18px 20px;margin-bottom:18px;
                    box-shadow:0 8px 32px rgba(238,121,39,0.3);
                    position:relative;overflow:hidden;'>
            <div style='position:absolute;top:-20px;right:-20px;width:80px;height:80px;
                        border-radius:50%;background:rgba(255,255,255,0.08);'></div>
            <div style='position:absolute;bottom:-30px;right:20px;width:100px;height:100px;
                        border-radius:50%;background:rgba(255,255,255,0.05);'></div>
            <div style='font-family:DM Serif Display,serif;font-size:20px;
                        color:#fff;position:relative;'>✦ AI Powered</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:10px;
                        color:rgba(255,255,255,0.7);margin-top:3px;
                        letter-spacing:0.1em;position:relative;'>
                AMAZON NOVA · BEDROCK
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Quick questions
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:9px;"
        "letter-spacing:0.12em;color:rgba(232,230,224,0.3);margin-bottom:8px;'>"
        "QUICK QUESTIONS</div>", unsafe_allow_html=True)

    quick = {
        "Top complaints":    "What are customers most commonly complaining about?",
        "At-risk products":  "Which products need urgent attention and why?",
        "What they love":    "What do customers love most?",
        "Exec summary":      "Give me a full executive summary of the reviews.",
    }
    q_cols = st.columns(2)
    for i,(label, question) in enumerate(quick.items()):
        if q_cols[i%2].button(label, key=f"qq{i}", use_container_width=True):
            st.session_state["chat_history"].append({"role":"user","content":question})
            with st.spinner(""):
                try:
                    reply = bedrock_reply(question, data, st.session_state["chat_history"][:-1])
                except Exception as e:
                    reply = f"⚠️ {e}"
            st.session_state["chat_history"].append({"role":"assistant","content":reply})
            st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Chat history container
    chat_box = st.container(height=380)
    with chat_box:
        if not st.session_state["chat_history"]:
            st.markdown(
                "<div style='text-align:center;padding:48px 16px;'>"
                "<div style='font-size:28px;margin-bottom:12px;'>✦</div>"
                "<div style='font-family:DM Serif Display,serif;font-size:16px;"
                "color:rgba(232,230,224,0.5);'>Ask me anything about<br>your customer reviews</div>"
                "<div style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                "color:rgba(232,230,224,0.2);margin-top:8px;letter-spacing:0.1em;'>"
                "POWERED BY AMAZON NOVA</div>"
                "</div>",
                unsafe_allow_html=True)
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about your reviews...", key="chat_right")
    if user_input:
        st.session_state["chat_history"].append({"role":"user","content":user_input})
        with st.spinner(""):
            try:
                reply = bedrock_reply(user_input, data, st.session_state["chat_history"][:-1])
            except Exception as e:
                reply = f"⚠️ Bedrock error: {e}"
        st.session_state["chat_history"].append({"role":"assistant","content":reply})
        st.rerun()

    # Clear
    if st.session_state["chat_history"]:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("🗑  Clear chat", use_container_width=True, key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    # Footer
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:9px;"
        "color:rgba(232,230,224,0.2);text-align:center;margin-top:12px;"
        "letter-spacing:0.08em;'>RETAILIQ · QUANTUMQUIRKS</div>",
        unsafe_allow_html=True)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Retail IQ", page_icon="🛍️", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # Hero header — forced 72px with ID anchor to beat Streamlit overrides
    st.markdown("""
        <div id='retailiq-hero' style='margin-bottom:2px;padding-top:8px;line-height:1;'>
            <span id='retailiq-title-main'>Retail</span><span id='retailiq-title-iq'>IQ</span>
        </div>
        <div id='retailiq-sub'>SENTIMENT &amp; INTENT ANALYTICS · POWERED BY QUANTUMQUIRKS</div>
        <style>
            #retailiq-title-main {
                font-family: 'DM Serif Display', serif !important;
                font-size: 72px !important;
                font-weight: 400 !important;
                color: #e8e6e0 !important;
                line-height: 1 !important;
                display: inline !important;
            }
            #retailiq-title-iq {
                font-family: 'DM Serif Display', serif !important;
                font-size: 72px !important;
                font-style: italic !important;
                color: #ee7927 !important;
                line-height: 1 !important;
                display: inline !important;
            }
            #retailiq-sub {
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 11px !important;
                letter-spacing: 0.14em !important;
                color: rgba(232,230,224,0.4) !important;
                margin-bottom: 28px !important;
                margin-top: 6px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Load data
    try:
        data = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error("`cleaned_reviews.csv` not found.")
        return
    except Exception as e:
        st.error(f"Load failed: {e}")
        return

    missing = validate_data(data)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    text_col = get_text_col(data)
    filters  = render_sidebar(data)
    filtered = apply_filters(data, filters["rr"], filters["eff_s"],
                              filters["eff_i"], filters["sel_d"], filters["age_r"])

    # ── Main layout: chat col declared first so it renders at the top ──────────
    main_col, chat_col = st.columns([3, 1], gap="large", vertical_alignment="top")

    # Write chat FIRST — Streamlit renders columns top-aligned when
    # the shorter column is populated before the taller one.
    with chat_col:
        render_chat_panel(filtered)

    with main_col:
        t1,t2,t3,t4 = st.tabs(["Overview","Product Insights","Review Explorer","Model Insights"])
        with t1: tab_overview(filtered, data, filters["top_n"])
        with t2: tab_product(filtered, data, text_col)
        with t3: tab_explorer(filtered, text_col)
        with t4: tab_model(text_col, filtered)


if __name__ == "__main__":
    main()
