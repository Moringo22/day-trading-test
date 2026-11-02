#!/usr/bin/env python3
"""
Test de day trading (papier). Une exécution :
  1. télécharge les bougies 5 min (Yahoo Finance via yfinance) et les ajoute à data/bars/
  2. recalcule indicateurs, chandelles, signaux et trades simulés (backtest + test en direct)
  3. écrit trades.csv et docs/data.json (lu par le site docs/index.html)

  python run.py            # exécution normale (GitHub Actions)
  python run.py --offline  # sans téléchargement (tests)
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dt import engine, model

ROOT = Path(__file__).parent
BARS = ROOT / "data" / "bars"
ET = ZoneInfo("America/New_York")
MTL = ZoneInfo("America/Toronto")


def load_cfg():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def log(msg):
    line = f"{datetime.now(MTL):%Y-%m-%d %H:%M} {msg}"
    print(line)
    with open(ROOT / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- données
_ALL = None


def _load_all():
    """Bougies stockées par jour : data/bars/AAAA-MM-JJ.csv (tous les titres). Un fichier par jour = petits commits."""
    global _ALL
    if _ALL is None:
        files = sorted(BARS.glob("*.csv"))
        if not files:
            _ALL = pd.DataFrame(columns=["ts", "ticker", "open", "high", "low", "close", "volume"])
        else:
            _ALL = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        _ALL["ts"] = pd.to_datetime(_ALL["ts"], utc=True).dt.tz_convert(ET)
    return _ALL


def load_bars(t):
    a = _load_all()
    df = a[a["ticker"] == t].drop(columns="ticker").set_index("ts").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    return df[~df.index.duplicated(keep="last")]


def save_new(frames):
    """frames: {ticker: df} -> réécrit seulement les fichiers des jours touchés."""
    global _ALL
    parts = [f.assign(ticker=t) for t, f in frames.items() if len(f)]
    if not parts:
        return
    new = pd.concat(parts).reset_index()
    new = new.rename(columns={new.columns[0]: "ts"})
    a = pd.concat([_load_all(), new], ignore_index=True)
    a = a.drop_duplicates(subset=["ts", "ticker"], keep="last").sort_values(["ts", "ticker"])
    _ALL = a
    BARS.mkdir(parents=True, exist_ok=True)
    days = set(new["ts"].dt.date)
    a = a.assign(day=a["ts"].dt.date)
    for d in days:
        part = a[a["day"] == d].drop(columns="day").copy()
        part["ts"] = part["ts"].map(lambda x: x.isoformat())
        part[["ts", "ticker", "open", "high", "low", "close", "volume"]].to_csv(BARS / f"{d}.csv", index=False)


def clean(df, now_et):
    df = df.dropna(subset=["open", "high", "low", "close"])
    tod = df.index.hour * 60 + df.index.minute
    df = df[(tod >= engine.OPEN_MIN) & (tod <= engine.LAST_BAR_MIN) & (df.index.dayofweek < 5)]
    df = df[df.index + timedelta(minutes=5) <= now_et]  # enlève la bougie en formation
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch(cfg, now_et):
    import yfinance as yf
    have = {t: load_bars(t) for t in cfg["tickers"]}
    period = "5d" if all(len(v) > 2000 for v in have.values()) else "59d"
    data = None
    for attempt in range(3):
        try:
            data = yf.download(cfg["tickers"], period=period, interval="5m", group_by="ticker",
                               auto_adjust=False, prepost=False, progress=False, threads=True)
            if data is not None and len(data):
                break
        except Exception as e:  # noqa: BLE001
            log(f"Téléchargement : essai {attempt + 1} échoué ({e})")
        time.sleep(10)
    if data is None or not len(data):
        log("ERREUR : aucune donnée reçue de Yahoo.")
        return 0
    added, fresh = 0, {}
    for t in cfg["tickers"]:
        try:
            d = data[t].copy()
        except KeyError:
            log(f"{t} : absent de la réponse")
            continue
        d.columns = [c.lower() for c in d.columns]
        d.index = pd.to_datetime(d.index)
        d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index
        d.index = d.index.tz_convert(ET)
        d = clean(d[["open", "high", "low", "close", "volume"]], now_et)
        old = have[t]
        added += len(d.index.difference(old.index))
        fresh[t] = d
    save_new(fresh)
    return added


# ----------------------------------------------------------------------------- calcul
def compute(cfg, now_et):
    frames = {t: load_bars(t) for t in cfg["tickers"]}
    frames = {t: engine.add_indicators(clean(f, now_et)) for t, f in frames.items() if len(f) > 100}
    if cfg["market_ticker"] not in frames:
        raise SystemExit("Pas de données pour l'indice de marché.")
    bias = engine.market_bias(frames[cfg["market_ticker"]])
    today = now_et.date()
    after_close = now_et.hour * 60 + now_et.minute >= 16 * 60 + 5
    fwd = pd.Timestamp(cfg["forward_start"]).date()

    trades = []
    for t, df in frames.items():
        for s in engine.find_signals(t, df, bias):
            d = s["signal_time"].date()
            complete = d < today or after_close
            tr = engine.simulate(s, df, cfg, complete)
            if tr is None:
                continue
            tr["periode"] = "test en direct" if d >= fwd else "backtest"
            trades.append(tr)
    model.score_trades(trades, cfg)
    return frames, trades


# ----------------------------------------------------------------------------- sorties
def iso(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else pd.Timestamp(x).isoformat()


def chart_time(ts):
    """Lightweight-charts affiche l'UTC : on décale pour afficher l'heure de New York/Montréal."""
    ts = pd.Timestamp(ts)
    return int(ts.timestamp() + ts.utcoffset().total_seconds())


def trade_row(t):
    keys = ["ticker", "strategy", "direction", "periode", "status", "score", "tier", "patterns", "rsi", "adx",
            "rvol", "ref_price", "entry", "stop", "target", "target_R", "shares", "exit", "exit_reason",
            "pnl_usd", "risk_usd", "R", "p_win", "ev_R", "n_bucket", "pris_filtre", "raison_filtre", "last_price"]
    r = {k: t.get(k) for k in keys}
    for k in ("rsi", "adx", "rvol"):
        r[k] = round(r[k], 2) if r[k] is not None else None
    r["signal_time"] = iso(t["signal_time"])
    r["entry_time"] = iso(t.get("entry_time"))
    r["exit_time"] = iso(t.get("exit_time"))
    r["reasons"] = t.get("reasons", [])
    return r


def verdict(s):
    n = s.get("n", 0)
    if n < 30:
        return {"code": "tot", "texte": f"Trop tôt : {n} trades fermés (minimum 30, idéalement 100+)."}
    if s["avg_R"] > 0.10 and s["ci_low"] is not None and s["ci_low"] > 0 and n >= 100:
        return {"code": "ok", "texte": "Avantage probable : moyenne > +0,10 R et intervalle de confiance au-dessus de 0."}
    if s["ci_high"] is not None and s["ci_high"] < 0:
        return {"code": "non", "texte": "Pas rentable : même le haut de l'intervalle de confiance est sous 0."}
    return {"code": "incertain", "texte": "Pas encore prouvé : l'intervalle de confiance inclut 0."}


def write_outputs(cfg, frames, trades, now_et, added):
    rows = [trade_row(t) for t in trades]
    pd.DataFrame(rows).to_csv(ROOT / "trades.csv", index=False)

    def sel(periode, filtre):
        return [t for t in trades if t["periode"] == periode and (t["pris_filtre"] or not filtre)]

    summary, equity, by_strat, calib = {}, {}, {}, {}
    for per in ("backtest", "test en direct"):
        summary[per] = {"tous": model.summarize(sel(per, False), cfg), "filtre": model.summarize(sel(per, True), cfg)}
        equity[per] = {"tous": model.equity_curve(sel(per, False), cfg), "filtre": model.equity_curve(sel(per, True), cfg)}
        by_strat[per] = model.breakdown(sel(per, False), cfg, lambda t: f"{t['strategy']} · {'achat' if t['direction']=='long' else 'vente à découvert'}")
        calib[per] = model.calibration(sel(per, False))
    by_tier = model.breakdown(sel("test en direct", False) or sel("backtest", False), cfg, lambda t: t["tier"])
    by_ticker = model.breakdown(trades, cfg, lambda t: t["ticker"])
    by_pattern = model.breakdown([t for t in trades if t["strategy"] == "REVERS"], cfg,
                                 lambda t: " + ".join(engine.PATTERN_FR[p] for p in t["patterns"].split(",") if p) or "—")

    today = now_et.date()
    last_day = max(df.index[-1].date() for df in frames.values())
    pistes = [trade_row(t) for t in trades if t["signal_time"].date() == last_day]
    pistes.sort(key=lambda r: r["signal_time"], reverse=True)

    charts = {}
    for tk, df in frames.items():
        days = sorted(set(df["date"]))[-2:]
        d = df[df["date"].isin(days)]
        charts[tk] = {
            "bars": [[chart_time(ix), round(r.open, 3), round(r.high, 3), round(r.low, 3), round(r.close, 3), int(r.volume)]
                     for ix, r in d.iterrows()],
            "vwap": [[chart_time(ix), round(v, 3)] for ix, v in d["vwap"].items()],
            "ema21": [[chart_time(ix), round(v, 3)] for ix, v in d["ema21"].items()],
            "trades": [{"t": chart_time(t["entry_time"]), "x": chart_time(t["exit_time"]) if t.get("exit_time") is not None else None,
                        "dir": t["direction"], "strat": t["strategy"], "R": t.get("R"), "status": t["status"],
                        "pris": t["pris_filtre"], "entry": t.get("entry"), "stop": t["stop"], "target": t.get("target")}
                       for t in trades if t["ticker"] == tk and t.get("entry_time") is not None
                       and t["signal_time"].date() in days],
        }

    tod = now_et.hour * 60 + now_et.minute
    out = {
        "generated_at": now_et.astimezone(MTL).isoformat(),
        "market_open": now_et.weekday() < 5 and engine.OPEN_MIN <= tod < 16 * 60,
        "last_bar_day": last_day.isoformat(), "bars_added": added,
        "config": {k: cfg[k] for k in ("tickers", "forward_start", "account_usd", "risk_per_trade_usd",
                                        "slippage_pct", "max_leverage", "min_ev_R", "daily_loss_limit_R", "max_open_filtered")},
        "strategies": {k: {"nom": v["nom"], "target_R": v["target_R"]} for k, v in engine.STRATS.items()},
        "summary": summary, "equity": equity, "by_strategy": by_strat, "by_tier": by_tier,
        "by_ticker": by_ticker, "by_pattern": by_pattern, "calibration": calib,
        "verdict": verdict(summary["test en direct"]["filtre"]),
        "pistes": pistes,
        "trades": [trade_row(t) for t in sorted(trades, key=lambda t: t["signal_time"], reverse=True)[:400]],
        "charts": charts,
        "log": (ROOT / "run_log.txt").read_text(encoding="utf-8").splitlines()[-15:] if (ROOT / "run_log.txt").exists() else [],
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "data.json").write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")


def main():
    cfg = load_cfg()
    now_et = datetime.now(ET)
    if "--now" in sys.argv:
        now_et = datetime.fromisoformat(sys.argv[sys.argv.index("--now") + 1]).astimezone(ET)
    added = 0 if "--offline" in sys.argv else fetch(cfg, now_et)
    frames, trades = compute(cfg, now_et)
    write_outputs(cfg, frames, trades, now_et, added)
    fwd = [t for t in trades if t["periode"] == "test en direct"]
    log(f"OK : {added} bougies ajoutées, {len(trades)} trades simulés ({len(fwd)} en direct, "
        f"{sum(t['pris_filtre'] for t in fwd)} pris par le modèle).")


if __name__ == "__main__":
    main()
