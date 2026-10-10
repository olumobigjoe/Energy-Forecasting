"""
Kajola Village Hybrid Microgrid - SoC & Energy Forecast Platform
Hybrid: First-order Differencing + BiLSTM (sequence branch) + ANN (exogenous branch)

Run:  streamlit run app.py
Deps: pip install streamlit pandas numpy plotly scikit-learn tensorflow
      (tensorflow is optional - an sklearn MLP fallback is used if it is missing)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models

    TF_OK = True
except Exception:  # TensorFlow not installed
    TF_OK = False

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Kajola Energy Forecast", page_icon="🔋", layout="wide")
st.title("🔋 Kajola Village Hybrid Microgrid - SoC & Energy Forecast Platform")
st.caption(
    "Hybrid Differencing + BiLSTM + ANN  |  Train on historical data → forecast future SoC and "
    "available energy.  Energy (kWh) = SoC (%) × Battery capacity (kWh) / 100"
)

SEQ_COLS = ["dsoc", "load", "solar", "hs", "hc", "soc"]
NEXT_COLS = ["load", "solar", "hs", "hc"]


# ----------------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dataset(start="2024-01-01", periods=17544, capacity=240.0, seed=42):
    """Synthetic Kajola-like hourly dataset (load, solar, battery SoC). Demo only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="h")
    h = idx.hour.values
    doy = idx.dayofyear.values

    weekend = (idx.dayofweek.values >= 5) * 1.0
    load = (
        14
        + 15 * np.exp(-((h - 20) ** 2) / 8)
        + 7 * np.exp(-((h - 7) ** 2) / 6)
        + 2 * weekend
        + rng.normal(0, 1.5, periods)
    )
    load = np.clip(load, 4, None)

    season = 1 - 0.25 * np.cos(2 * np.pi * (doy - 20) / 365)  # rainy-season dip
    n_days = periods // 24 + 2
    cloud_day = np.repeat(rng.uniform(0.45, 1.0, n_days), 24)[:periods]
    clear = np.maximum(0, np.sin((h - 6) / 12 * np.pi))
    solar = np.clip(95 * clear * season * cloud_day + rng.normal(0, 1.5, periods) * (clear > 0), 0, None)

    e = np.empty(periods)
    e[0] = 0.6 * capacity
    lo, hi = 0.05 * capacity, capacity
    for t in range(1, periods):
        net = solar[t] - load[t]
        e[t] = e[t - 1] + (net * 0.95 if net > 0 else net / 0.95)
        e[t] = min(max(e[t], lo), hi)
    soc = e / capacity * 100
    return pd.DataFrame({"load": load, "solar": solar, "soc": soc}, index=idx)


def prepare_df(raw, tcol, lcol, scol, soc_col, soc_fraction):
    df = pd.DataFrame(
        {
            "load": pd.to_numeric(raw[lcol], errors="coerce"),
            "solar": pd.to_numeric(raw[scol], errors="coerce"),
            "soc": pd.to_numeric(raw[soc_col], errors="coerce"),
        }
    )
    df.index = pd.to_datetime(raw[tcol], errors="coerce")
    df = df[df.index.notna()]
    df = df[~df.index.duplicated()].sort_index()
    df = df.asfreq("h").interpolate(limit_direction="both")
    if soc_fraction:
        df["soc"] = df["soc"] * 100
    df["soc"] = df["soc"].clip(0, 100)
    return df


def build_frame(df):
    d = df.copy()
    d["hs"] = np.sin(2 * np.pi * d.index.hour / 24)
    d["hc"] = np.cos(2 * np.pi * d.index.hour / 24)
    d["dsoc"] = d["soc"].diff()  # <-- differencing step
    return d.dropna()


def make_windows(d, L):
    S = d[SEQ_COLS].values
    n = len(S)
    W = sliding_window_view(S, (L, S.shape[1]))[:, 0][:-1]  # (n-L, L, F)
    nxt = d[NEXT_COLS].values[L:]
    soc_i = d["soc"].values[L - 1 : n - 1]
    exo = np.column_stack([nxt, soc_i])
    y = d["dsoc"].values[L:]
    return W, exo, y, soc_i, d.index[L:]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def train_model(df, cfg, progress):
    d = build_frame(df)
    L = cfg["lookback"]
    n_tr_rows = int(len(d) * cfg["train_frac"])

    sc_s, sc_e, sc_y = StandardScaler(), StandardScaler(), StandardScaler()
    S_scaled = sc_s.fit(d[SEQ_COLS].values[:n_tr_rows]).transform(d[SEQ_COLS].values)
    d_scaled = d.copy()
    d_scaled[SEQ_COLS] = S_scaled

    W, _, _, _, _ = make_windows(d_scaled, L)
    _, exo, y, soc_prev, tidx = make_windows(d, L)
    n_samples = len(y)
    n_tr = int(n_samples * cfg["train_frac"])

    exo_s = sc_e.fit(exo[:n_tr]).transform(exo)
    y_s = sc_y.fit(y[:n_tr].reshape(-1, 1)).transform(y.reshape(-1, 1)).ravel()

    Wtr, Wte = W[:n_tr], W[n_tr:]
    Etr, Ete = exo_s[:n_tr], exo_s[n_tr:]
    ytr = y_s[:n_tr]

    np.random.seed(cfg["seed"])
    history = {}

    if TF_OK:
        tf.random.set_seed(cfg["seed"])
        seq_in = layers.Input((L, W.shape[2]))
        x = layers.Bidirectional(layers.LSTM(cfg["units"]))(seq_in)
        x = layers.Dropout(0.1)(x)
        ex_in = layers.Input((exo.shape[1],))
        e = layers.Dense(32, activation="relu")(ex_in)
        e = layers.Dense(16, activation="relu")(e)
        c = layers.Concatenate()([x, e])
        c = layers.Dense(32, activation="relu")(c)
        out = layers.Dense(1)(c)
        model = models.Model([seq_in, ex_in], out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

        class StProgress(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                progress.progress(
                    (epoch + 1) / cfg["epochs"],
                    text=f"Epoch {epoch + 1}/{cfg['epochs']} - loss {logs.get('loss', 0):.4f} "
                    f"| val {logs.get('val_loss', 0):.4f}",
                )

        hist = model.fit(
            [Wtr, Etr],
            ytr,
            epochs=cfg["epochs"],
            batch_size=cfg["batch"],
            validation_split=0.1,
            shuffle=False,
            verbose=0,
            callbacks=[
                StProgress(),
                callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            ],
        )
        history = {"loss": hist.history["loss"], "val_loss": hist.history["val_loss"]}

        def predict(Wx, Ex):
            return model([Wx.astype("float32"), Ex.astype("float32")], training=False).numpy().ravel()

        engine = "BiLSTM + ANN (TensorFlow/Keras)"
    else:
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=cfg["epochs"],
            early_stopping=True,
            random_state=cfg["seed"],
        )
        mlp.fit(np.hstack([Wtr.reshape(len(Wtr), -1), Etr]), ytr)
        history = {"loss": mlp.loss_curve_, "val_loss": None}
        progress.progress(1.0, text="Training finished (sklearn MLP fallback)")

        def predict(Wx, Ex):
            return mlp.predict(np.hstack([Wx.reshape(len(Wx), -1), Ex]))

        engine = "MLP fallback (install TensorFlow for BiLSTM)"

    # --- one-step-ahead evaluation on the held-out (chronological) test set
    d_pred = sc_y.inverse_transform(predict(Wte, Ete).reshape(-1, 1)).ravel()
    soc_true = soc_prev[n_tr:] + y[n_tr:]
    soc_hat = np.clip(soc_prev[n_tr:] + d_pred, 0, 100)
    soc_naive = soc_prev[n_tr:]  # persistence baseline

    def m(a, b):
        return {
            "RMSE (%)": float(np.sqrt(mean_squared_error(a, b))),
            "MAE (%)": float(mean_absolute_error(a, b)),
            "R²": float(r2_score(a, b)),
        }

    return {
        "predict": predict,
        "sc_s": sc_s,
        "sc_e": sc_e,
        "sc_y": sc_y,
        "cfg": cfg,
        "history": history,
        "engine": engine,
        "test_idx": tidx[n_tr:],
        "soc_true": soc_true,
        "soc_hat": soc_hat,
        "metrics": {"Hybrid model": m(soc_true, soc_hat), "Persistence baseline": m(soc_true, soc_naive)},
        "sigma": float(np.std(soc_true - soc_hat)),
    }


def forecast_future(bundle, df, horizon, load_scale, solar_scale, soc_floor):
    cfg = bundle["cfg"]
    L = cfg["lookback"]
    d = build_frame(df)

    prof = d.assign(month=d.index.month, hour=d.index.hour)
    by_mh = prof.groupby(["month", "hour"])[["load", "solar"]].mean()
    by_h = prof.groupby("hour")[["load", "solar"]].mean()

    buf = [list(r) for r in d[SEQ_COLS].values[-L:]]
    soc = float(d["soc"].iloc[-1])
    t = d.index[-1]
    rows = []
    for _ in range(horizon):
        t = t + pd.Timedelta(hours=1)
        key = (t.month, t.hour)
        base = by_mh.loc[key] if key in by_mh.index else by_h.loc[t.hour]
        load = float(base["load"]) * load_scale
        solar = float(base["solar"]) * solar_scale
        hs, hc = np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24)

        seq = bundle["sc_s"].transform(np.array(buf[-L:]))[None]
        exo = bundle["sc_e"].transform(np.array([[load, solar, hs, hc, soc]]))
        d_s = bundle["predict"](seq, exo).reshape(-1, 1)
        dsoc = float(bundle["sc_y"].inverse_transform(d_s)[0, 0])

        new_soc = float(np.clip(soc + dsoc, soc_floor, 100))
        buf.append([new_soc - soc, load, solar, hs, hc, new_soc])
        soc = new_soc
        rows.append((t, load, solar, soc))

    out = pd.DataFrame(rows, columns=["Timestamp", "Load kW", "Solar kW", "Forecast SoC %"])
    k = np.arange(1, horizon + 1)
    band = 1.96 * bundle["sigma"] * np.sqrt(k)  # indicative only
    out["Lower %"] = np.clip(out["Forecast SoC %"] - band, 0, 100)
    out["Upper %"] = np.clip(out["Forecast SoC %"] + band, 0, 100)
    return out


# ----------------------------------------------------------------------------
# Sidebar - configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    capacity = st.number_input("Battery capacity (kWh)", 10.0, 5000.0, 240.0, 10.0)

    st.subheader("Model")
    lookback = st.slider("Look-back window (hours)", 6, 72, 24, 6)
    units = st.slider("BiLSTM units", 8, 128, 32, 8)
    epochs = st.slider("Max epochs", 5, 100, 25, 5)
    batch = st.select_slider("Batch size", [32, 64, 128, 256], value=128)
    train_frac = st.slider("Train fraction", 0.5, 0.9, 0.8, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 42)
    st.caption(f"Engine: {'TensorFlow BiLSTM + ANN' if TF_OK else 'sklearn MLP fallback (TensorFlow not found)'}")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset", "🧠 Train Model", "📈 Evaluation", "🔮 Forecast"]
)

# ---- Tab 1: dataset ---------------------------------------------------------
with tab1:
    st.subheader("Dataset")
    src = st.radio("Data source", ["Synthetic demo (Kajola-like)", "Upload CSV"], horizontal=True)
    df = None

    if src.startswith("Synthetic"):
        df = generate_dataset(capacity=capacity)
        st.info("Using a **synthetic** 17,544-hour dataset (2024–2025) for demonstration. "
                "Upload your real measurements for real results.")
    else:
        up = st.file_uploader("Hourly CSV with timestamp, load (kW), solar (kW), SoC", type="csv")
        if up is not None:
            raw = pd.read_csv(up)
            cols = list(raw.columns)
            c1, c2, c3, c4 = st.columns(4)
            tcol = c1.selectbox("Timestamp", cols, index=0)
            lcol = c2.selectbox("Load kW", cols, index=min(1, len(cols) - 1))
            scol = c3.selectbox("Solar kW", cols, index=min(2, len(cols) - 1))
            soc_col = c4.selectbox("SoC", cols, index=min(3, len(cols) - 1))
            frac = st.checkbox("SoC is a fraction (0-1) rather than percent")
            try:
                df = prepare_df(raw, tcol, lcol, scol, soc_col, frac)
            except Exception as ex:
                st.error(f"Could not parse the file: {ex}")
        else:
            st.warning("Upload a CSV to continue.")

    if df is not None:
        st.session_state["df"] = df
        a, b, c, d_, e_ = st.columns(5)
        a.metric("Samples", f"{len(df):,}")
        b.metric("Start", df.index[0].strftime("%Y-%m-%d"))
        c.metric("End", df.index[-1].strftime("%Y-%m-%d"))
        d_.metric("Mean load", f"{df['load'].mean():.1f} kW")
        e_.metric("Mean SoC", f"{df['soc'].mean():.1f} %")

        rng = st.date_input(
            "View range",
            (df.index[-1].date() - pd.Timedelta(days=7), df.index[-1].date()),
            min_value=df.index[0].date(),
            max_value=df.index[-1].date(),
        )
        if isinstance(rng, tuple) and len(rng) == 2:
            view = df.loc[str(rng[0]) : str(rng[1])]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=view.index, y=view["load"], name="Load kW"))
            fig.add_trace(go.Scatter(x=view.index, y=view["solar"], name="Solar kW"))
            fig.add_trace(go.Scatter(x=view.index, y=view["soc"], name="SoC %", yaxis="y2"))
            fig.update_layout(
                yaxis=dict(title="kW"),
                yaxis2=dict(title="SoC %", overlaying="y", side="right", range=[0, 100]),
                hovermode="x unified",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Preview data / download"):
            st.dataframe(df.head(200), use_container_width=True)
            st.download_button(
                "Download dataset CSV",
                df.rename_axis("timestamp").reset_index().to_csv(index=False),
                "kajola_dataset.csv",
            )

# ---- Tab 2: training --------------------------------------------------------
with tab2:
    st.subheader("Train hybrid model")
    st.markdown(
        "1. **Differencing**: target = ΔSoC(t+1) = SoC(t+1) − SoC(t), making the series stationary.\n"
        "2. **BiLSTM** reads the last *N* hours of [ΔSoC, load, solar, time, SoC].\n"
        "3. **ANN** reads next-hour load/solar/time + current SoC.\n"
        "4. Branches are concatenated → predicted ΔSoC → added back to SoC."
    )
    if "df" not in st.session_state:
        st.warning("Load a dataset in the first tab.")
    elif st.button("🚀 Train model", type="primary"):
        cfg = dict(lookback=lookback, units=units, epochs=epochs, batch=batch,
                   train_frac=train_frac, seed=int(seed))
        bar = st.progress(0.0, text="Preparing…")
        with st.spinner("Training…"):
            st.session_state["bundle"] = train_model(st.session_state["df"], cfg, bar)
        st.success("Training complete. See the Evaluation tab.")

    if "bundle" in st.session_state:
        h = st.session_state["bundle"]["history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=h["loss"], name="Training loss"))
        if h.get("val_loss"):
            fig.add_trace(go.Scatter(y=h["val_loss"], name="Validation loss"))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="MSE (scaled)", height=350,
                          title=f"Training curve - {st.session_state['bundle']['engine']}")
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: evaluation ------------------------------------------------------
with tab3:
    st.subheader("Held-out test performance (one-step-ahead SoC)")
    if "bundle" not in st.session_state:
        st.info("Train a model first.")
    else:
        B = st.session_state["bundle"]
        mt = pd.DataFrame(B["metrics"]).T
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE SoC", f"{mt.loc['Hybrid model', 'RMSE (%)']:.3f} %")
        c2.metric("MAE SoC", f"{mt.loc['Hybrid model', 'MAE (%)']:.3f} %")
        c3.metric("R²", f"{mt.loc['Hybrid model', 'R²']:.4f}")
        st.dataframe(mt.style.format("{:.4f}"), use_container_width=True)
        st.caption("Compare against the persistence baseline (SoC(t+1) = SoC(t)); a useful model should beat it.")

        n_show = st.slider("Hours to display", 24, min(1000, len(B["soc_true"])), 168)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=B["test_idx"][:n_show], y=B["soc_true"][:n_show], name="Actual"))
        fig.add_trace(go.Scatter(x=B["test_idx"][:n_show], y=B["soc_hat"][:n_show], name="Predicted"))
        fig.update_layout(yaxis_title="SoC %", height=380, hovermode="x unified",
                          title="Actual vs Predicted SoC")
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        sc = go.Figure(go.Scatter(x=B["soc_true"], y=B["soc_hat"], mode="markers",
                                  marker=dict(size=3, opacity=0.5), name="points"))
        lim = [float(np.min(B["soc_true"])), float(np.max(B["soc_true"]))]
        sc.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal", line=dict(dash="dash")))
        sc.update_layout(title="Predicted vs Actual", xaxis_title="Actual %", yaxis_title="Predicted %", height=350)
        col_a.plotly_chart(sc, use_container_width=True)
        res = B["soc_true"] - B["soc_hat"]
        hist = go.Figure(go.Histogram(x=res, nbinsx=50))
        hist.update_layout(title="Residuals (Actual − Predicted)", xaxis_title="SoC %", height=350)
        col_b.plotly_chart(hist, use_container_width=True)

# ---- Tab 4: forecast --------------------------------------------------------
with tab4:
    st.subheader("Future SoC & energy forecast")
    if "bundle" not in st.session_state:
        st.info("Train a model first.")
    else:
        df = st.session_state["df"]
        c1, c2, c3, c4 = st.columns(4)
        horizon = c1.slider("Horizon (hours)", 24, 168, 24, 24)
        load_scale = c2.slider("Load scenario (× historical)", 0.5, 2.0, 1.0, 0.05)
        solar_scale = c3.slider("Solar scenario (× historical)", 0.3, 1.5, 1.0, 0.05)
        thresh = c4.slider("Low-SoC alert threshold (%)", 5, 50, 20)
        soc_floor = st.slider("Battery minimum SoC (%) - protection limit", 0, 30, 5)
        st.caption(f"Forecast starts after the last data point ({df.index[-1]}). "
                   "Future load/solar come from the historical month-hour profile × your scenario factors.")

        if st.button("🔮 Run forecast", type="primary"):
            with st.spinner("Forecasting recursively…"):
                fc = forecast_future(st.session_state["bundle"], df, horizon, load_scale, solar_scale, soc_floor)
            fc["Available Energy kWh"] = fc["Forecast SoC %"] / 100 * capacity
            st.session_state["fc"] = fc

        if "fc" in st.session_state:
            fc = st.session_state["fc"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean SoC", f"{fc['Forecast SoC %'].mean():.2f} %")
            m2.metric("Min SoC", f"{fc['Forecast SoC %'].min():.2f} %")
            m3.metric("Mean energy", f"{fc['Available Energy kWh'].mean():.1f} kWh")
            m4.metric("Min energy", f"{fc['Available Energy kWh'].min():.1f} kWh")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fc["Timestamp"], y=fc["Upper %"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fc["Timestamp"], y=fc["Lower %"], fill="tonexty",
                                     line=dict(width=0), name="~95% indicative band"))
            fig.add_trace(go.Scatter(x=fc["Timestamp"], y=fc["Forecast SoC %"],
                                     mode="lines+markers", name="Forecast SoC %"))
            fig.add_hline(y=thresh, line_dash="dash", annotation_text="Alert threshold")
            fig.update_layout(yaxis=dict(title="SoC %", range=[0, 100]), height=420, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=fc["Timestamp"], y=fc["Load kW"], name="Load kW"))
            fig2.add_trace(go.Scatter(x=fc["Timestamp"], y=fc["Solar kW"], name="Solar kW"))
            fig2.update_layout(height=300, title="Scenario inputs used", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

            below = fc[fc["Forecast SoC %"] < thresh]
            if below.empty:
                st.success(f"SoC stays above {thresh}% for the whole horizon.")
            else:
                st.error(
                    f"SoC falls below {thresh}% in {len(below)} of {len(fc)} hours "
                    f"(first at {below['Timestamp'].iloc[0]}). Consider a stronger charging strategy, "
                    "load shifting, or backup generation."
                )

            st.dataframe(fc.round(2), use_container_width=True)
            st.download_button("Download forecast CSV", fc.round(3).to_csv(index=False), "soc_forecast.csv")
            st.markdown("**Formula:** Energy (kWh) = SoC (%) × Capacity (kWh) / 100 - SoC forecasting *is* energy forecasting.")
            st.caption("Recursive multi-step forecasts accumulate error; the shaded band is a rough "
                       "indicator (test-set error σ × √h), not a calibrated prediction interval.")
