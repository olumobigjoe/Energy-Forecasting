"""
core.py - logic for the Kajola Microgrid SoC & Energy Forecast Platform (no Streamlit here).

Pipeline: hourly data -> battery SoC simulation -> first-order differencing (dSoC)
          -> BiLSTM (sequence branch) + ANN (exogenous branch) -> dSoC_hat -> SoC -> Energy (kWh)
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models

    TF_OK = True
except Exception:  # TensorFlow not installed -> sklearn fallback
    TF_OK = False

SEQ_COLS = ["dsoc", "load", "solar", "temp", "hum", "hs", "hc", "soc"]
EXO_COLS = ["load", "solar", "temp", "hum", "hs", "hc"]  # next-hour inputs (+ current soc)


# ---------------------------------------------------------------- data ----
def _find(cols, *cands):
    for cand in cands:
        for c in cols:
            if cand in str(c).lower():
                return c
    return None


def load_hourly(src):
    """Read the hourly CSV (path or file-like). Handles duplicate load columns and a missing SoC."""
    raw = pd.read_csv(src)
    cols = list(raw.columns)
    tcol = _find(cols, "timestamp", "datetime", "date", "time") or cols[0]
    lcol = _find(cols, "load")
    scol = _find(cols, "solar gen", "pv", "solar")
    if lcol is None or scol is None:
        raise ValueError("CSV needs a load column and a solar generation column.")
    tmp, hum = _find(cols, "temp"), _find(cols, "humid")
    irr, soc = _find(cols, "irradiance"), _find(cols, "soc")

    df = pd.DataFrame(
        {
            "load": pd.to_numeric(raw[lcol], errors="coerce").values,
            "solar": pd.to_numeric(raw[scol], errors="coerce").values,
            "temp": pd.to_numeric(raw[tmp], errors="coerce").values if tmp else 0.0,
            "hum": pd.to_numeric(raw[hum], errors="coerce").values if hum else 0.0,
        },
        index=pd.to_datetime(raw[tcol].values, errors="coerce"),
    )
    if irr:
        df["irr"] = pd.to_numeric(raw[irr], errors="coerce").values
    if soc:
        df["soc"] = pd.to_numeric(raw[soc], errors="coerce").values
    df = df[df.index.notna()]
    df = df[~df.index.duplicated()].sort_index().asfreq("h").interpolate(limit_direction="both")
    return df, soc is not None


def simulate_soc(load, solar, capacity, soc_min, soc_init, eff):
    """Energy-balance battery model. Surplus charges (x eff), deficit discharges (/ eff)."""
    lo, e = capacity * soc_min / 100, capacity * soc_init / 100
    out = np.empty(len(load))
    for i, (l, s) in enumerate(zip(load, solar)):
        n = s - l
        e += n * eff if n > 0 else n / eff
        e = min(max(e, lo), capacity)
        out[i] = e / capacity * 100
    return out


def with_soc(df, capacity, soc_min, soc_init, eff, has_soc):
    df = df.copy()
    if not has_soc:
        df["soc"] = simulate_soc(df["load"].values, df["solar"].values, capacity, soc_min, soc_init, eff)
    df["soc"] = df["soc"].clip(0, 100)
    return df


def build_frame(df):
    d = df.copy()
    d["hs"] = np.sin(2 * np.pi * d.index.hour / 24)
    d["hc"] = np.cos(2 * np.pi * d.index.hour / 24)
    d["dsoc"] = d["soc"].diff()  # differencing step
    return d.dropna(subset=SEQ_COLS)


def make_windows(d, L):
    S = d[SEQ_COLS].values
    n = len(S)
    W = sliding_window_view(S, (L, S.shape[1]))[:, 0][:-1]
    exo = np.column_stack([d[EXO_COLS].values[L:], d["soc"].values[L - 1 : n - 1]])
    y = d["dsoc"].values[L:]
    return W, exo, y, d["soc"].values[L - 1 : n - 1], d.index[L:]


# --------------------------------------------------------------- model ----
def train_model(df, cfg, on_epoch=None):
    d = build_frame(df)
    L = cfg["lookback"]
    n_tr_rows = int(len(d) * cfg["train_frac"])

    sc_s = StandardScaler().fit(d[SEQ_COLS].values[:n_tr_rows])
    ds = d.copy()
    ds[SEQ_COLS] = sc_s.transform(d[SEQ_COLS].values)
    W = make_windows(ds, L)[0]
    _, exo, y, soc_prev, tidx = make_windows(d, L)
    n_tr = int(len(y) * cfg["train_frac"])

    sc_e = StandardScaler().fit(exo[:n_tr])
    sc_y = StandardScaler().fit(y[:n_tr].reshape(-1, 1))
    exo_s = sc_e.transform(exo)
    y_s = sc_y.transform(y.reshape(-1, 1)).ravel()
    Wtr, Wte, Etr, Ete, ytr = W[:n_tr], W[n_tr:], exo_s[:n_tr], exo_s[n_tr:], y_s[:n_tr]

    np.random.seed(cfg["seed"])
    if TF_OK:
        tf.random.set_seed(cfg["seed"])
        seq_in = layers.Input((L, W.shape[2]))
        x = layers.Dropout(0.1)(layers.Bidirectional(layers.LSTM(cfg["units"]))(seq_in))
        ex_in = layers.Input((exo.shape[1],))
        e = layers.Dense(16, activation="relu")(layers.Dense(32, activation="relu")(ex_in))
        c = layers.Dense(32, activation="relu")(layers.Concatenate()([x, e]))
        model = models.Model([seq_in, ex_in], layers.Dense(1)(c))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

        class Cb(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if on_epoch:
                    on_epoch(epoch, cfg["epochs"], logs or {})

        h = model.fit(
            [Wtr, Etr], ytr, epochs=cfg["epochs"], batch_size=cfg["batch"], validation_split=0.1,
            shuffle=False, verbose=0,
            callbacks=[Cb(), callbacks.EarlyStopping(patience=6, restore_best_weights=True)],
        )
        history = {"loss": h.history["loss"], "val_loss": h.history["val_loss"]}

        def predict(Wx, Ex):
            return model([Wx.astype("float32"), Ex.astype("float32")], training=False).numpy().ravel()

        engine = "BiLSTM + ANN (TensorFlow)"
    else:
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=cfg["epochs"] * 8, early_stopping=True, random_state=cfg["seed"])
        mlp.fit(np.hstack([Wtr.reshape(len(Wtr), -1), Etr]), ytr)
        history = {"loss": list(mlp.loss_curve_), "val_loss": None}
        if on_epoch:
            on_epoch(cfg["epochs"] - 1, cfg["epochs"], {})

        def predict(Wx, Ex):
            return mlp.predict(np.hstack([Wx.reshape(len(Wx), -1), Ex]))

        engine = "MLP fallback (install TensorFlow for BiLSTM)"

    b = {"predict": predict, "sc_s": sc_s, "sc_e": sc_e, "sc_y": sc_y, "cfg": cfg,
         "history": history, "engine": engine, "n_train": n_tr, "n_test": len(y) - n_tr}

    # one-step-ahead evaluation
    dp = sc_y.inverse_transform(predict(Wte, Ete).reshape(-1, 1)).ravel()
    true = soc_prev[n_tr:] + y[n_tr:]
    hat = np.clip(soc_prev[n_tr:] + dp, cfg["soc_min"], 100)
    base = soc_prev[n_tr:]
    b.update(test_idx=tidx[n_tr:], soc_true=true, soc_hat=hat)

    # 24-h recursive backtest on the test period
    n = len(d)
    first = L + n_tr - 1
    origins = np.unique(np.linspace(first, n - 25, 15).astype(int))
    errs, base_errs = [], []
    for p in origins:
        buf = d[SEQ_COLS].values[p - L + 1 : p + 1]
        exo_f = d[["load", "solar", "temp", "hum"]].values[p + 1 : p + 25]
        pred = rollout(b, buf, d["soc"].values[p], d.index[p + 1 : p + 25], exo_f, cfg["soc_min"])
        truth = d["soc"].values[p + 1 : p + 25]
        errs.append(truth - pred)
        base_errs.append(truth - d["soc"].values[p])
    errs, base_errs = np.array(errs), np.array(base_errs)

    def m(a, p):
        return {"RMSE (%)": float(np.sqrt(mean_squared_error(a, p))),
                "MAE (%)": float(mean_absolute_error(a, p)), "R²": float(r2_score(a, p))}

    rm = lambda e: {"RMSE (%)": float(np.sqrt((e ** 2).mean())), "MAE (%)": float(np.abs(e).mean()), "R²": float("nan")}
    b["metrics"] = {
        "Hybrid · 1-step ahead": m(true, hat),
        "Persistence · 1-step": m(true, base),
        "Hybrid · 24 h recursive": rm(errs),
        "Persistence · 24 h": rm(base_errs),
    }
    b["sigma_h"] = np.maximum(np.sqrt((errs ** 2).mean(axis=0)), 0.02)  # RMSE per horizon step
    return b


def rollout(b, buf, soc, times, exo, soc_floor):
    """Recursive multi-step SoC prediction. exo rows = [load, solar, temp, hum]."""
    L = b["cfg"]["lookback"]
    buf = [list(r) for r in buf]
    out = []
    for t, (load, solar, temp, hum) in zip(times, exo):
        hs, hc = np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24)
        seq = b["sc_s"].transform(np.asarray(buf[-L:]))[None]
        ex = b["sc_e"].transform([[load, solar, temp, hum, hs, hc, soc]])
        dsoc = float(b["sc_y"].inverse_transform(b["predict"](seq, ex).reshape(-1, 1))[0, 0])
        new = float(np.clip(soc + dsoc, soc_floor, 100))
        buf.append([new - soc, load, solar, temp, hum, hs, hc, new])
        soc = new
        out.append(new)
    return np.array(out)


def forecast(b, df, horizon, load_scale, solar_scale, batt):
    """Forecast `horizon` hours after the last data point using month-hour profiles x scenario factors."""
    d = build_frame(df)
    L = b["cfg"]["lookback"]
    d2 = d.assign(month=d.index.month, hour=d.index.hour)
    cols = ["load", "solar", "temp", "hum"]
    by_mh = d2.groupby(["month", "hour"])[cols].mean()
    by_h = d2.groupby("hour")[cols].mean()

    times = pd.date_range(d.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    exo = np.array([(by_mh.loc[(t.month, t.hour)] if (t.month, t.hour) in by_mh.index else by_h.loc[t.hour]).values
                    for t in times])
    exo[:, 0] *= load_scale
    exo[:, 1] *= solar_scale

    last = float(d["soc"].iloc[-1])
    ml = rollout(b, d[SEQ_COLS].values[-L:], last, times, exo, batt["soc_min"])
    phys = simulate_soc(exo[:, 0], exo[:, 1], batt["capacity"], batt["soc_min"], last, batt["eff"])

    sig = b["sigma_h"][np.minimum(np.arange(horizon), len(b["sigma_h"]) - 1)]
    out = pd.DataFrame({
        "Timestamp": times, "Load kW": exo[:, 0], "Solar kW": exo[:, 1], "Temp °C": exo[:, 2],
        "Humidity %": exo[:, 3], "Forecast SoC %": ml, "Energy-balance SoC %": phys,
        "Lower %": np.clip(ml - 1.96 * sig, 0, 100), "Upper %": np.clip(ml + 1.96 * sig, 0, 100),
    })
    out["Available Energy kWh"] = out["Forecast SoC %"] / 100 * batt["capacity"]
    return out
