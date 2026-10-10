"""
Kajola Microgrid SoC & Energy Forecast Platform  -  Final Year Project
Run:  streamlit run app.py      (keep core.py and the CSV files in the same folder)
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core

# ============================ EDIT YOUR PROJECT DETAILS HERE ============================
PROJECT = dict(
    title="Kajola Microgrid SoC & Energy Forecast",
    student="Your Name",
    matric="Matric No.",
    supervisor="Supervisor Name",
    dept="Department of Electrical & Electronic Engineering",
    school="Your University",
    year="2026",
)
# ========================================================================================

st.set_page_config(page_title="Kajola Energy Forecast", page_icon="🔋", layout="wide")

APP_DIR = Path(__file__).parent
DEFAULT_DATA = APP_DIR / "kajola_2024_2025_hourly.csv"
DEFAULT_REF = APP_DIR / "forecast_2026_2027_24h.csv"

CYAN, GREEN, AMBER, RED, VIOLET = "#38bdf8", "#22d3a6", "#f5b945", "#ff5c7a", "#a78bfa"
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

# ------------------------------------------------------------------ styling ----
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }

/* ---------- background: lighter navy with soft teal / indigo glows ---------- */
.stApp { color: #e8eefb; background-attachment: fixed;
  background: radial-gradient(900px 520px at 10% -6%, rgba(34,211,166,.20), transparent 62%),
              radial-gradient(900px 560px at 96% 2%, rgba(99,102,241,.28), transparent 62%),
              linear-gradient(180deg, #14213f 0%, #0e1830 55%, #0b1326 100%); background-attachment: fixed; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1400px; }

/* ---------- global text: bright and readable ---------- */
.stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p { color: #e8eefb; }
.stApp h1, .stApp h2, .stApp h3 { color: #ffffff; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #b4c3dc !important; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #172647 0%, #0f1a34 100%); border-right: 1px solid rgba(255,255,255,.12); }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #f3f7ff !important; font-weight: 500; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #b4c3dc !important; }
[data-testid="stExpander"] details { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.16); border-radius: 14px; }
[data-testid="stExpander"] summary { background: transparent !important; border-radius: 14px; }
[data-testid="stExpander"] summary:hover { background: rgba(255,255,255,.08) !important; }
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color: #ffffff !important; font-weight: 600; }
[data-testid="stExpander"] summary svg { fill: #ffffff; color: #ffffff; }

/* ---------- inputs, selects, sliders, uploader ---------- */
div[data-baseweb="input"], div[data-baseweb="input"] > div, div[data-baseweb="base-input"], div[data-baseweb="select"] > div
  { background: #1c2d52 !important; border-color: rgba(255,255,255,.24) !important; }
input, textarea { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: #ffffff; }
[data-testid="stNumberInput"] button { background: #2a4170 !important; color: #ffffff !important; }
[data-testid="stNumberInput"] button svg { fill: #ffffff; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"] { background: #1c2d52 !important; color: #ffffff !important; }
div[data-baseweb="popover"] li:hover { background: #2f4a82 !important; }
[data-testid="stSliderThumbValue"], [data-testid="stThumbValue"],
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] { color: #22d3a6 !important; font-weight: 700; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #b4c3dc !important; }
div[data-baseweb="slider"] [role="slider"] { background: #22d3a6 !important; box-shadow: 0 0 0 4px rgba(34,211,166,.28); }
div[data-baseweb="slider"] div[style*="255, 75, 75"] { filter: hue-rotate(150deg) saturate(1.2); }
[data-testid="stFileUploaderDropzone"] { background: #1c2d52; border: 1px dashed rgba(255,255,255,.35); border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] * { color: #e8eefb !important; }
[data-testid="stFileUploaderDropzone"] button { background: #2a4170; border: 1px solid rgba(255,255,255,.3); }

/* ---------- hero ---------- */
.hero { position: relative; overflow: hidden; border-radius: 24px; padding: 30px 36px; margin-bottom: 18px;
        background: linear-gradient(120deg, rgba(34,211,166,.26), rgba(56,189,248,.20) 55%, rgba(167,139,250,.24));
        border: 1px solid rgba(255,255,255,.2); box-shadow: 0 20px 50px rgba(0,0,0,.3); }
.hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
              background: radial-gradient(circle, rgba(34,211,166,.35), transparent 70%); }
.hero h1 { margin: 10px 0 6px; font-size: 2.3rem; font-weight: 800; color: #ffffff; letter-spacing: -.02em; }
.hero p { color: #dbe6f8; font-size: 1.04rem; margin: 0 0 12px; max-width: 900px; }
.chips span { display: inline-block; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
              border-radius: 999px; padding: 4px 13px; margin: 0 8px 6px 0; font-size: .78rem; font-weight: 600; color: #f3f7ff; }
.meta { font-size: .84rem; color: #c3d1e8; margin-top: 6px; }

/* ---------- cards ---------- */
.kpi { background: linear-gradient(180deg, rgba(255,255,255,.11), rgba(255,255,255,.04)); border: 1px solid rgba(255,255,255,.14);
       border-top: 3px solid var(--a); border-radius: 16px; padding: 15px 18px; height: 100%; box-shadow: 0 8px 24px rgba(0,0,0,.22); }
.kl { font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; color: #c3d1e8; font-weight: 700; }
.kv { font-size: 1.85rem; font-weight: 800; color: #ffffff; line-height: 1.25; margin-top: 4px; }
.ks { font-size: .8rem; color: #b4c3dc; margin-top: 2px; }
.card { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); border-radius: 18px; padding: 18px 22px; margin-bottom: 12px; }
.card h4 { margin: 0 0 6px; color: #ffffff; font-size: 1.04rem; }
.card p, .card li { color: #dbe6f8; font-size: .93rem; line-height: 1.6; }
.banner { border-radius: 16px; padding: 16px 22px; margin: 6px 0 14px; border: 1px solid; }
.banner b { font-size: 1.05rem; color: #ffffff; } .banner div { color: #e8eefb; font-size: .93rem; margin-top: 4px; }
.crit { background: rgba(255,92,122,.16); border-color: rgba(255,92,122,.65); }
.warn { background: rgba(245,185,69,.15); border-color: rgba(245,185,69,.65); }
.ok   { background: rgba(34,211,166,.15); border-color: rgba(34,211,166,.65); }
.sec { font-size: 1.25rem; font-weight: 700; color: #ffffff; margin: 22px 0 8px; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"], [data-testid="stTabs"] [role="tablist"] { gap: 6px; background: rgba(255,255,255,.08); padding: 6px; border-radius: 16px; }
.stTabs [data-baseweb="tab"], [data-testid="stTabs"] button[role="tab"] { border-radius: 11px; padding: 9px 20px; height: auto; }
.stTabs [data-baseweb="tab"] p, [data-testid="stTabs"] button[role="tab"] p { color: #e3ebf9 !important; font-weight: 600; font-size: .98rem; }
.stTabs [aria-selected="true"], [data-testid="stTabs"] button[aria-selected="true"] { background: linear-gradient(135deg, #22d3a6, #38bdf8) !important; }
.stTabs [aria-selected="true"] p, [data-testid="stTabs"] button[aria-selected="true"] p { color: #04101a !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; background: transparent !important; }

/* ---------- PREDICT button ---------- */
div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #22d3a6 0%, #38bdf8 100%); border: 0; border-radius: 18px; padding: 1rem 1.6rem;
  animation: pulse 2.2s infinite; transition: transform .15s; }
div.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {
  color: #04101a !important; font-size: 1.3rem; font-weight: 800; letter-spacing: .16em; }
div.stButton > button[kind="primary"]:hover { transform: translateY(-2px) scale(1.01); }
div.stButton > button:not([kind="primary"]), [data-testid="stBaseButton-secondary"] { background: #22365f; border: 1px solid rgba(255,255,255,.28); }
div.stButton > button:not([kind="primary"]) p, [data-testid="stBaseButton-secondary"] p { color: #ffffff !important; font-weight: 600; }
[data-testid="stDownloadButton"] button { background: #22365f; border: 1px solid rgba(255,255,255,.28); }
[data-testid="stDownloadButton"] button p { color: #ffffff !important; font-weight: 600; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,211,166,.55); } 70% { box-shadow: 0 0 0 20px rgba(34,211,166,0); }
                   100% { box-shadow: 0 0 0 0 rgba(34,211,166,0); } }

/* ---------- method pipeline ---------- */
.pipe { display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; margin: 8px 0 16px; }
.step { flex: 1; min-width: 150px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
        border-radius: 14px; padding: 12px 14px; font-size: .85rem; color: #dbe6f8; }
.step b { display: block; color: #ffffff; margin-bottom: 3px; font-size: .93rem; }
.arr { align-self: center; color: #38bdf8; font-size: 1.4rem; }
</style>
"""


def html(s):
    st.markdown(" ".join(l.strip() for l in s.strip().splitlines()), unsafe_allow_html=True)


def kpi(col, label, value, sub="", accent=GREEN):
    col.markdown(
        f'<div class="kpi" style="--a:{accent}"><div class="kl">{label}</div>'
        f'<div class="kv">{value}</div><div class="ks">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style(fig, h=380, title=None):
    fig.update_layout(
        template="plotly_dark", height=h + (20 if title else 0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e3ebf9", size=13), margin=dict(l=10, r=10, t=95 if title else 45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=13, color="#e3ebf9")),
        hovermode="x unified",
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, y=0.97, yanchor="top", font=dict(size=17, color="#ffffff")))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.10)", tickfont=dict(color="#cfdaee"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,.10)", tickfont=dict(color="#cfdaee"))
    return fig


def plot(fig):
    st.plotly_chart(fig, **STRETCH)


def gauge(value, title, rng, suffix, color, steps):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": suffix, "font": {"size": 34, "color": "#fff"}},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": rng}, "bar": {"color": color}, "bgcolor": "rgba(255,255,255,.05)",
               "borderwidth": 0, "steps": steps}))
    fig.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#e3ebf9"),
                      margin=dict(l=25, r=25, t=45, b=5))
    return fig


html(CSS)

# ----------------------------------------------------------------- sidebar ----
with st.sidebar:
    html('<div style="font-size:1.25rem;font-weight:800;color:#fff;">🔋 Kajola Forecast</div>'
         '<div style="color:#8da2c0;font-size:.8rem;margin-bottom:10px;">Hybrid Differencing + BiLSTM + ANN</div>')
    with st.expander("🔋 Battery system", expanded=True):
        capacity = st.number_input("Capacity (kWh)", 10.0, 5000.0, 240.0, 10.0)
        soc_min = st.slider("Minimum SoC / protection floor (%)", 0, 30, 10)
        soc_init = st.slider("Initial SoC (%)", 0, 100, 50)
        eff = st.slider("Charge / discharge efficiency", 0.80, 1.00, 0.95, 0.01)
    with st.expander("🧠 Model settings"):
        lookback = st.slider("Look-back window (h)", 6, 72, 24, 6)
        units = st.slider("BiLSTM units", 8, 128, 32, 8)
        epochs = st.slider("Max epochs", 5, 100, 30, 5)
        batch = st.select_slider("Batch size", [32, 64, 128, 256], value=128)
        train_frac = st.slider("Train fraction", 0.5, 0.9, 0.8, 0.05)
        seed = st.number_input("Random seed", 0, 9999, 42)
    with st.expander("📁 Data files"):
        up_data = st.file_uploader("Hourly dataset (CSV)", type="csv")
        up_ref = st.file_uploader("Thesis forecast (CSV, optional)", type="csv")
    st.caption(f"Engine: {'TensorFlow BiLSTM + ANN' if core.TF_OK else 'sklearn MLP fallback (TensorFlow not installed)'}")


# -------------------------------------------------------------------- data ----
@st.cache_data(show_spinner="Reading dataset…")
def read_hourly(raw: bytes):
    return core.load_hourly(io.BytesIO(raw))


@st.cache_data(show_spinner="Simulating battery SoC…")
def derive(df, has, cap, smin, sinit, e):
    return core.with_soc(df, cap, smin, sinit, e, has)


if up_data is not None:
    raw_bytes = up_data.getvalue()
elif DEFAULT_DATA.exists():
    raw_bytes = DEFAULT_DATA.read_bytes()
else:
    st.error("Dataset not found. Put `kajola_2024_2025_hourly.csv` next to app.py or upload it in the sidebar.")
    st.stop()

try:
    df0, has_soc = read_hourly(raw_bytes)
except Exception as ex:
    st.error(f"Could not read the dataset: {ex}")
    st.stop()
df = derive(df0, has_soc, capacity, soc_min, soc_init, eff)

ref = None
try:
    rb = up_ref.getvalue() if up_ref is not None else (DEFAULT_REF.read_bytes() if DEFAULT_REF.exists() else None)
    if rb:
        r = pd.read_csv(io.BytesIO(rb))
        ref = pd.DataFrame({"Timestamp": pd.to_datetime(r.iloc[:, 0]), "Thesis SoC %": pd.to_numeric(r.iloc[:, 1])})
except Exception:
    ref = None

batt = dict(capacity=capacity, soc_min=soc_min, eff=eff)
cov = df.solar.sum() / df.load.sum() * 100
deficit_day = (df.load.mean() - df.solar.mean()) * 24
floor_share = (df.soc <= soc_min + 0.05).mean() * 100


def cfg_now():
    return dict(lookback=lookback, units=units, epochs=epochs, batch=batch, train_frac=train_frac,
                seed=int(seed), soc_min=soc_min)


def model_sig():
    return (len(df), round(float(df.load.sum()), 2), round(float(df.soc.sum()), 2), tuple(cfg_now().items()))


def get_model(force=False):
    b = st.session_state.get("bundle")
    if b is not None and b["sig"] == model_sig() and not force:
        return b
    bar = st.progress(0.0, text="Preparing training windows…")

    def cb(ep, total, logs):
        bar.progress(min((ep + 1) / total, 1.0),
                     text=f"Training epoch {ep + 1}/{total} · loss {logs.get('loss', 0):.4f} · val {logs.get('val_loss', 0):.4f}")

    b = core.train_model(df, cfg_now(), cb)
    b["sig"] = model_sig()
    bar.empty()
    st.session_state["bundle"] = b
    return b


# -------------------------------------------------------------------- hero ----
P = PROJECT
html(f"""
<div class="hero">
  <div class="chips"><span>⚡ Hybrid Differencing + BiLSTM + ANN</span><span>📍 Kajola, Sagamu · 6.8°N 3.6°E</span>
  <span>🕒 {len(df):,} hourly samples</span><span>🔋 {capacity:.0f} kWh battery</span></div>
  <h1>🔋 {P['title']}</h1>
  <p>Forecasting battery State-of-Charge and available energy for a hybrid solar microgrid.
  Trained on {df.index[0]:%Y}–{df.index[-1]:%Y} data, predicting {df.index[-1].year + 1} and beyond.</p>
  <div class="meta">{P['student']} · {P['matric']} &nbsp;|&nbsp; Supervisor: {P['supervisor']} &nbsp;|&nbsp; {P['dept']}, {P['school']} · {P['year']}</div>
</div>""")

tab_ov, tab_data, tab_model, tab_pred, tab_about = st.tabs(
    ["🏠 Overview", "📊 Data Explorer", "🧠 Model Lab", "🔮 PREDICT", "ℹ️ About"])

# ----------------------------------------------------------------- overview ----
with tab_ov:
    c = st.columns(6)
    kpi(c[0], "Samples", f"{len(df):,}", "hourly records", CYAN)
    kpi(c[1], "Period", f"{df.index[0]:%b %Y} – {df.index[-1]:%b %Y}", f"{(df.index[-1] - df.index[0]).days + 1} days", CYAN)
    kpi(c[2], "Avg load", f"{df.load.mean():.1f} kW", f"peak {df.load.max():.0f} kW", AMBER)
    kpi(c[3], "Avg solar", f"{df.solar.mean():.1f} kW", f"peak {df.solar.max():.0f} kW", AMBER)
    kpi(c[4], "PV coverage", f"{cov:.1f}%", "of total demand", RED if cov < 50 else GREEN)
    kpi(c[5], "At SoC floor", f"{floor_share:.1f}%", f"of hours at {soc_min}%", RED if floor_share > 50 else GREEN)

    need = df.load.sum() / df.solar.sum()
    cls = "crit" if floor_share > 50 else "ok"
    html(f"""<div class="banner {cls}"><b>{'⚠️ Energy-deficient system' if cls == 'crit' else '✅ Balanced system'}</b>
    <div>Solar supplies only <b>{cov:.1f}%</b> of demand — an average deficit of <b>{deficit_day:,.0f} kWh/day</b> against a
    {capacity:.0f} kWh battery, so the battery sits at its {soc_min}% floor {floor_share:.1f}% of the time.
    Matching annual demand would need roughly <b>{need:.1f}×</b> today's PV output (≈{df.solar.max() * need:,.0f} kW peak,
    before storage losses).</div></div>""")

    a, b_ = st.columns(2)
    m = df.resample("MS")[["load", "solar"]].mean() * 24
    fig = go.Figure()
    fig.add_bar(x=m.index, y=m.load, name="Load", marker_color=AMBER)
    fig.add_bar(x=m.index, y=m.solar, name="Solar", marker_color=GREEN)
    fig.update_layout(barmode="group")
    a.plotly_chart(style(fig, 360, "Average daily energy by month (kWh/day)"), **STRETCH)

    prof = df.groupby(df.index.hour)[["load", "solar"]].mean()
    fig = go.Figure()
    fig.add_scatter(x=prof.index, y=prof.load, name="Load", line=dict(color=AMBER, width=3))
    fig.add_scatter(x=prof.index, y=prof.solar, name="Solar", fill="tozeroy", line=dict(color=GREEN, width=3),
                    fillcolor="rgba(34,211,166,.18)")
    fig.update_xaxes(title="Hour of day", dtick=3)
    b_.plotly_chart(style(fig, 360, "Average daily profile (kW)"), **STRETCH)

# ------------------------------------------------------------ data explorer ----
with tab_data:
    names = {"Load (kW)": "load", "Solar generation (kW)": "solar", "Temperature (°C)": "temp",
             "Humidity (%)": "hum", "Irradiance (W/m²)": "irr", "Battery SoC (%)": "soc"}
    names = {k: v for k, v in names.items() if v in df}
    c1, c2 = st.columns([2, 1])
    picks = c1.multiselect("Variables", list(names), default=["Load (kW)", "Solar generation (kW)", "Battery SoC (%)"])
    end = df.index[-1].date()
    rng = c2.date_input("Date range", ((df.index[-1] - pd.Timedelta(days=6)).date(), end),
                        min_value=df.index[0].date(), max_value=end)
    if picks and isinstance(rng, tuple) and len(rng) == 2:
        view = df.loc[str(rng[0]): str(rng[1])]
        fig = make_subplots(rows=len(picks), cols=1, shared_xaxes=True, vertical_spacing=0.04)
        colors = [CYAN, AMBER, GREEN, VIOLET, RED, "#e879f9"]
        for i, p in enumerate(picks):
            fig.add_scatter(x=view.index, y=view[names[p]], name=p, line=dict(color=colors[i % 6], width=2), row=i + 1, col=1)
            fig.update_yaxes(title_text=p.split(" (")[0], row=i + 1, col=1)
        plot(style(fig, 160 * len(picks) + 80))

    a, b_ = st.columns(2)
    pv = df.assign(h=df.index.hour, m=df.index.month).pivot_table(index="h", columns="m", values="load")
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale="Tealgrn", colorbar=dict(title="kW")))
    fig.update_xaxes(title="Month", dtick=1)
    fig.update_yaxes(title="Hour")
    a.plotly_chart(style(fig, 380, "Average load: hour × month").update_layout(hovermode="closest"), **STRETCH)
    cc = df[[v for v in ["load", "solar", "temp", "hum", "irr"] if v in df]].corr()
    fig = go.Figure(go.Heatmap(z=cc.values, x=cc.columns, y=cc.columns, zmid=0, colorscale="RdBu",
                               text=np.round(cc.values, 2), texttemplate="%{text}"))
    b_.plotly_chart(style(fig, 380, "Correlation matrix").update_layout(hovermode="closest"), **STRETCH)

    with st.expander("Preview & download dataset (with derived SoC)"):
        st.dataframe(df.reset_index(names="Timestamp").head(300), hide_index=True, **STRETCH)
        st.download_button("⬇ Download dataset CSV", df.reset_index(names="Timestamp").to_csv(index=False),
                           "kajola_with_soc.csv")
    if not has_soc:
        st.caption("ℹ️ The file has no SoC column, so SoC was derived with the energy-balance battery model "
                   "(capacity, floor, initial SoC and efficiency are set in the sidebar).")

# --------------------------------------------------------------- model lab ----
with tab_model:
    html("""<div class="pipe">
    <div class="step"><b>1 · Differencing</b>Target = ΔSoC(t+1) = SoC(t+1) − SoC(t)</div><div class="arr">→</div>
    <div class="step"><b>2a · BiLSTM branch</b>Reads the last N hours of load, solar, weather, time, SoC</div><div class="arr">+</div>
    <div class="step"><b>2b · ANN branch</b>Reads next-hour load, solar, weather, time + current SoC</div><div class="arr">→</div>
    <div class="step"><b>3 · Fusion</b>Concatenate → dense layers → predicted ΔSoC</div><div class="arr">→</div>
    <div class="step"><b>4 · Reconstruct</b>SoC = SoC + ΔSoC → Energy = SoC × Capacity / 100</div></div>""")

    if st.button("🔁 Train / retrain model"):
        with st.spinner("Training…"):
            get_model(force=True)

    B = st.session_state.get("bundle")
    if B is None:
        st.info("No model trained yet — press **Train / retrain model** here, or just press **PREDICT** (it trains automatically).")
    else:
        if B["sig"] != model_sig():
            st.warning("Settings changed since this model was trained. Retrain, or press PREDICT to retrain automatically.")
        M = B["metrics"]
        c = st.columns(5)
        kpi(c[0], "RMSE · 24 h ahead", f"{M['Hybrid · 24 h recursive']['RMSE (%)']:.3f}%", "recursive backtest", GREEN)
        kpi(c[1], "RMSE · 1 h ahead", f"{M['Hybrid · 1-step ahead']['RMSE (%)']:.3f}%", "held-out test set", GREEN)
        kpi(c[2], "MAE · 1 h ahead", f"{M['Hybrid · 1-step ahead']['MAE (%)']:.3f}%", "SoC points", CYAN)
        kpi(c[3], "Train / test", f"{B['n_train']:,} / {B['n_test']:,}", "chronological split", CYAN)
        kpi(c[4], "Engine", "BiLSTM+ANN" if core.TF_OK else "MLP", "TensorFlow" if core.TF_OK else "sklearn fallback", AMBER)

        st.markdown('<div class="sec">Benchmark against a naive baseline</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(M).T.round(4), **STRETCH)
        st.caption(f"The battery is at its protection floor {floor_share:.1f}% of the time, so a naive 'SoC stays the same' "
                   "baseline is already very strong on this dataset. The comparison is shown for transparency.")

        a, b_ = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(y=B["history"]["loss"], name="Training loss", line=dict(color=CYAN, width=3))
        if B["history"].get("val_loss"):
            fig.add_scatter(y=B["history"]["val_loss"], name="Validation loss", line=dict(color=AMBER, width=3))
        fig.update_xaxes(title="Epoch")
        a.plotly_chart(style(fig, 330, "Training convergence"), **STRETCH)
        fig = go.Figure(go.Bar(x=np.arange(1, len(B["sigma_h"]) + 1), y=B["sigma_h"], marker_color=VIOLET))
        fig.update_xaxes(title="Forecast lead time (h)")
        fig.update_yaxes(title="RMSE (SoC %)")
        b_.plotly_chart(style(fig, 330, "Error growth with lead time"), **STRETCH)

        n_show = st.slider("Hours of test period to display", 24, min(1000, len(B["soc_true"])), 168)
        fig = go.Figure()
        fig.add_scatter(x=B["test_idx"][:n_show], y=B["soc_true"][:n_show], name="Actual", line=dict(color=GREEN, width=2))
        fig.add_scatter(x=B["test_idx"][:n_show], y=B["soc_hat"][:n_show], name="Predicted", line=dict(color=CYAN, width=2, dash="dot"))
        fig.update_yaxes(title="SoC %")
        plot(style(fig, 360, "Actual vs predicted SoC (test set)"))

# ----------------------------------------------------------------- PREDICT ----
with tab_pred:
    html("""<div class="card"><h4>🔮 Generate a forecast</h4><p>Choose a horizon and scenario, then press
    <b>PREDICT</b>. The platform trains the hybrid model on first use, forecasts SoC hour by hour after the last
    recorded timestamp, and converts it to available energy (kWh). An independent energy-balance simulation is plotted
    alongside as a physical cross-check.</p></div>""")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        horizon = c1.select_slider("Forecast horizon (hours)", [24, 48, 72, 120, 168], value=24)
        load_scale = c2.slider("Load scenario (× historical)", 0.5, 2.0, 1.0, 0.05)
        solar_scale = c3.slider("Solar scenario (× historical)", 0.3, 4.0, 1.0, 0.1)
        thresh = c4.slider("Low-SoC alert level (%)", 5, 60, 20)
        if load_scale != 1.0 or solar_scale != 1.0:
            st.caption("⚠️ Scenario active: the model was trained on historical conditions, so the amber energy-balance "
                       "line is the more reliable guide when load/solar are changed.")

    _, mid, _ = st.columns([1, 1.4, 1])
    clicked = mid.button("⚡  PREDICT", type="primary", key="predict", **STRETCH)

    if clicked:
        with st.status("Running forecast pipeline…", expanded=True) as stt:
            st.write(f"① Dataset ready — {len(df):,} hourly samples, SoC derived")
            need_train = (st.session_state.get("bundle") or {}).get("sig") != model_sig()
            st.write("② Training the hybrid model…" if need_train else "② Trained model found ✔")
            B = get_model()
            st.write(f"③ Forecasting {horizon} h ahead from {df.index[-1] + pd.Timedelta(hours=1):%d %b %Y %H:%M}…")
            st.session_state["fc"] = core.forecast(B, df, horizon, load_scale, solar_scale, batt)
            stt.update(label="✅ Forecast ready", state="complete", expanded=False)

    fc = st.session_state.get("fc")
    if fc is None:
        html('<div class="card" style="text-align:center;padding:34px;"><h4>No forecast yet</h4>'
             '<p>Press <b>PREDICT</b> to generate one.</p></div>')
    else:
        soc = fc["Forecast SoC %"]
        mean_soc, min_soc, max_soc = soc.mean(), soc.min(), soc.max()
        mean_e = fc["Available Energy kWh"].mean()
        at_floor = int((soc <= soc_min + 0.5).sum())
        if mean_soc <= soc_min + 1:
            cls, head = "crit", "🔴 CRITICAL — battery pinned at its protection floor"
        elif mean_soc < thresh:
            cls, head = "warn", "🟡 LOW — state of charge below the alert level"
        else:
            cls, head = "ok", "🟢 HEALTHY — state of charge above the alert level"
        extra = "· enhanced charging strategy or extra generation required" if cls == "crit" else ""
        html(f"""<div class="banner {cls}"><b>{head}</b><div>Mean SoC <b>{mean_soc:.2f}%</b>
        ({mean_e:.1f} kWh of {capacity:.0f} kWh) · {at_floor} of {len(fc)} hours at the {soc_min}% floor ·
        forecast load {fc['Load kW'].mean():.0f} kW vs solar {fc['Solar kW'].mean():.0f} kW {extra}</div></div>""")

        g1, g2, g3 = st.columns([1, 1, 1.3])
        hi = max(thresh, soc_min + 6)
        steps = [{"range": [0, soc_min + 5], "color": "rgba(255,92,122,.28)"},
                 {"range": [soc_min + 5, hi], "color": "rgba(245,185,69,.22)"},
                 {"range": [hi, 100], "color": "rgba(34,211,166,.18)"}]
        g1.plotly_chart(gauge(mean_soc, "Mean forecast SoC", [0, 100], "%", CYAN, steps), **STRETCH)
        g2.plotly_chart(gauge(mean_e, "Mean available energy", [0, capacity], " kWh", GREEN,
                              [{"range": [0, capacity * .2], "color": "rgba(255,92,122,.28)"},
                               {"range": [capacity * .2, capacity], "color": "rgba(34,211,166,.15)"}]), **STRETCH)
        with g3:
            r1, r2 = st.columns(2)
            kpi(r1, "Min SoC", f"{min_soc:.2f}%", f"{fc['Available Energy kWh'].min():.1f} kWh", RED)
            kpi(r2, "Max SoC", f"{max_soc:.2f}%", f"{fc['Available Energy kWh'].max():.1f} kWh", GREEN)
            st.write("")
            r3, r4 = st.columns(2)
            kpi(r3, "Hours at floor", f"{at_floor}/{len(fc)}", f"≤ {soc_min + 0.5:.1f}% SoC", AMBER)
            kpi(r4, "Horizon", f"{len(fc)} h", f"{fc.Timestamp.iloc[0]:%d %b} → {fc.Timestamp.iloc[-1]:%d %b %Y}", CYAN)

        zoom = st.toggle("Zoom Y-axis to forecast range", True)
        t = fc["Timestamp"]
        fig = go.Figure()
        fig.add_scatter(x=t, y=fc["Upper %"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=t, y=fc["Lower %"], fill="tonexty", fillcolor="rgba(56,189,248,.16)", line=dict(width=0),
                        name="95% empirical band")
        fig.add_scatter(x=t, y=soc, name="Hybrid model forecast", mode="lines+markers", line=dict(color=CYAN, width=3),
                        marker=dict(size=6))
        fig.add_scatter(x=t, y=fc["Energy-balance SoC %"], name="Energy-balance check", line=dict(color=AMBER, width=2, dash="dash"))
        note = ""
        if ref is not None:
            mg = fc.merge(ref, on="Timestamp")
            if len(mg):
                fig.add_scatter(x=mg.Timestamp, y=mg["Thesis SoC %"], name="Thesis forecast (uploaded)",
                                line=dict(color=VIOLET, width=2, dash="dot"))
                diff = (mg["Forecast SoC %"] - mg["Thesis SoC %"]).abs().mean()
                note = f"Mean absolute difference vs uploaded thesis forecast: {diff:.3f} SoC points over {len(mg)} h."
        fig.add_hline(y=soc_min, line_dash="dot", line_color=RED, annotation_text=f"Protection floor {soc_min}%")
        fig.update_yaxes(title="SoC %", range=None if zoom else [0, 100])
        plot(style(fig, 420, "State-of-Charge forecast"))
        if note:
            st.caption(note)

        a, b_ = st.columns(2)
        fig = go.Figure(go.Scatter(x=t, y=fc["Available Energy kWh"], fill="tozeroy", line=dict(color=GREEN, width=3),
                                   fillcolor="rgba(34,211,166,.2)", name="Available energy"))
        fig.update_yaxes(title="kWh", range=None if zoom else [0, capacity])
        a.plotly_chart(style(fig, 330, "Available battery energy (kWh)"), **STRETCH)
        fig = go.Figure()
        fig.add_scatter(x=t, y=fc["Load kW"], name="Load", line=dict(color=AMBER, width=3))
        fig.add_scatter(x=t, y=fc["Solar kW"], name="Solar", fill="tozeroy", line=dict(color=GREEN, width=2),
                        fillcolor="rgba(34,211,166,.18)")
        fig.update_yaxes(title="kW")
        b_.plotly_chart(style(fig, 330, "Scenario inputs (month-hour profile × factors)"), **STRETCH)

        st.markdown('<div class="sec">Hourly forecast table</div>', unsafe_allow_html=True)
        tbl = fc[["Timestamp", "Forecast SoC %", "Available Energy kWh", "Energy-balance SoC %", "Load kW", "Solar kW"]].copy()
        tbl["Status"] = np.where(tbl["Forecast SoC %"] <= soc_min + 1, "🔴 Critical",
                                 np.where(tbl["Forecast SoC %"] < thresh, "🟡 Low", "🟢 Healthy"))
        st.dataframe(tbl.round(3), hide_index=True, **STRETCH, column_config={
            "Timestamp": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
            "Forecast SoC %": st.column_config.ProgressColumn("Forecast SoC %", min_value=0, max_value=100, format="%.2f"),
            "Available Energy kWh": st.column_config.NumberColumn(format="%.2f kWh"),
        })
        st.download_button("⬇ Download forecast CSV", fc.round(4).to_csv(index=False), "soc_energy_forecast.csv")
        st.caption("Recursive forecasts accumulate error with lead time; the band is the empirical RMSE per lead hour "
                   "from a rolling 24-h backtest. Energy (kWh) = SoC (%) × capacity / 100.")

# ------------------------------------------------------------------- about ----
with tab_about:
    a, b_ = st.columns(2)
    split = int(train_frac * 100)
    soc_note = "is provided in the file" if has_soc else (
        f"is derived from an energy-balance battery model ({capacity:.0f} kWh, {soc_min}% floor, {eff:.0%} efficiency)")
    with a:
        html(f"""<div class="card"><h4>🎯 Aim</h4><p>Forecast the battery State-of-Charge of the Kajola village hybrid
        microgrid so that available energy can be predicted ahead of time. Because
        <b>Energy (kWh) = SoC (%) × {capacity:.0f} kWh / 100</b>, forecasting SoC is forecasting energy.</p></div>
        <div class="card"><h4>🧪 Method</h4><ul>
        <li><b>Differencing</b> removes the level so the network learns hour-to-hour change.</li>
        <li><b>BiLSTM</b> captures temporal patterns in the last {lookback} hours.</li>
        <li><b>ANN</b> maps next-hour load, solar, weather and time to the change.</li>
        <li><b>Chronological split</b> ({split}% train / {100 - split}% test) avoids look-ahead leakage.</li>
        <li><b>Rolling 24-h backtest</b> measures real multi-step skill.</li></ul></div>""")
    with b_:
        html(f"""<div class="card"><h4>🗂 Data</h4><p>{len(df):,} hourly records ({df.index[0]:%d %b %Y} – {df.index[-1]:%d %b %Y}):
        load, solar generation, temperature, humidity and irradiance. SoC {soc_note}.</p></div>
        <div class="card"><h4>⚠️ Limitations (worth stating in your defence)</h4><ul>
        <li>The battery is at its floor {floor_share:.1f}% of the time, so SoC varies very little and errors are tiny in absolute terms.</li>
        <li>Future load/solar are month-hour averages × scenario factors, not weather forecasts.</li>
        <li>The model learns historical behaviour; large scenario changes are better judged with the energy-balance line.</li></ul></div>""")
    html(f'<div style="text-align:center;color:#6f84a3;font-size:.8rem;margin-top:18px;">{P["student"]} · {P["dept"]} · {P["school"]} · {P["year"]} — Final Year Project</div>')
