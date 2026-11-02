"""
Moteur du test de day trading (paper trading, aucun argent réel).

- Indicateurs : VWAP de séance, EMA 9/21, RSI 14, ATR 14, Bollinger 20/2, ADX 14,
  volume relatif (même tranche horaire des 10 séances précédentes), range d'ouverture.
- Chandelles : marteau, étoile filante, avalement haussier/baissier, étoile du matin/du soir, doji.
- 3 stratégies à règles fixes (aucune optimisation sur les données) :
    ORB      cassure du range d'ouverture (15 min) avec volume, cible 2R
    VWAP     repli sur le VWAP/EMA21 dans une tendance (ADX), cible 2R
    REVERS   chandelle de retournement aux extrêmes (RSI + Bollinger), cible 1,5R
- Simulation prudente : entrée à l'ouverture de la bougie suivante + glissement,
  si stop et cible sont touchés dans la même bougie on compte le stop,
  sortie forcée à la clôture de 15 h 55, frais IBKR inclus.
"""
import math
import numpy as np
import pandas as pd

OPEN_MIN, LAST_BAR_MIN = 9 * 60 + 30, 15 * 60 + 55

STRATS = {
    "ORB": {"nom": "Cassure du range d'ouverture", "target_R": 2.0, "start": 9 * 60 + 45, "end": 11 * 60},
    "VWAP": {"nom": "Repli sur VWAP en tendance", "target_R": 2.0, "start": 10 * 60, "end": 15 * 60},
    "REVERS": {"nom": "Chandelle de retournement", "target_R": 1.5, "start": 10 * 60, "end": 15 * 60},
}

PATTERN_FR = {
    "hammer": "Marteau", "shooting_star": "Étoile filante", "bull_engulf": "Avalement haussier",
    "bear_engulf": "Avalement baissier", "morning_star": "Étoile du matin", "evening_star": "Étoile du soir",
    "doji": "Doji",
}


# ----------------------------------------------------------------------------- indicateurs
def _wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df):
    df = df.copy()
    df["date"] = df.index.date
    df["tod"] = df.index.hour * 60 + df.index.minute
    h, l, c, o, v = df["high"], df["low"], df["close"], df["open"], df["volume"].astype(float)

    tp = (h + l + c) / 3
    pv = (tp * v).groupby(df["date"]).cumsum()
    cv = v.groupby(df["date"]).cumsum().replace(0, np.nan)
    df["vwap"] = (pv / cv).fillna(tp)

    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()

    d = c.diff()
    up, dn = _wilder(d.clip(lower=0), 14), _wilder((-d).clip(lower=0), 14)
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    df["rsi"] = df["rsi"].fillna(50)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = _wilder(tr, 14)

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = mid + 2 * sd, mid - 2 * sd

    pdm = h.diff()
    mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    atr_w = _wilder(tr, 14)
    pdi = 100 * _wilder(pdm, 14) / atr_w
    mdi = 100 * _wilder(mdm, 14) / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = _wilder(dx.fillna(0), 14)

    # Volume relatif : même tranche horaire, moyenne des 10 séances précédentes
    piv = df.pivot_table(index="date", columns="tod", values="volume", aggfunc="sum")
    hist = piv.rolling(10, min_periods=3).mean().shift(1)
    key = pd.MultiIndex.from_arrays([df["date"], df["tod"]])
    stacked = hist.stack()
    avg = pd.Series(stacked.reindex(key).values, index=df.index)
    fallback = v.rolling(20, min_periods=5).mean().shift(1)
    df["rvol"] = (v / avg.fillna(fallback)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Range d'ouverture (9 h 30 à 9 h 45)
    first = df[df["tod"] < OPEN_MIN + 15]
    orh = first.groupby("date")["high"].max()
    orl = first.groupby("date")["low"].min()
    df["or_high"] = df["date"].map(orh)
    df["or_low"] = df["date"].map(orl)

    last_close = c.groupby(df["date"]).last()
    prev_close = last_close.shift(1)
    df["prev_close"] = df["date"].map(prev_close)

    _patterns(df)
    return df


def _patterns(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    down_before = c.shift(1) < c.shift(4)
    up_before = c.shift(1) > c.shift(4)
    o1, c1, b1 = o.shift(1), c.shift(1), body.shift(1)
    o2, c2, b2 = o.shift(2), c.shift(2), body.shift(2)
    atr = df["atr"]

    df["p_hammer"] = (lower >= 2 * body) & (upper <= 0.25 * rng) & (rng > 0.3 * atr) & down_before
    df["p_shooting_star"] = (upper >= 2 * body) & (lower <= 0.25 * rng) & (rng > 0.3 * atr) & up_before
    df["p_bull_engulf"] = (c1 < o1) & (c > o) & (o <= c1) & (c >= o1) & (body > b1)
    df["p_bear_engulf"] = (c1 > o1) & (c < o) & (o >= c1) & (c <= o1) & (body > b1)
    df["p_morning_star"] = (c2 < o2) & (b2 > 0.6 * atr) & (b1 < 0.3 * b2) & (c > o) & (c > (o2 + c2) / 2)
    df["p_evening_star"] = (c2 > o2) & (b2 > 0.6 * atr) & (b1 < 0.3 * b2) & (c < o) & (c < (o2 + c2) / 2)
    df["p_doji"] = body <= 0.1 * rng
    for col in [x for x in df.columns if x.startswith("p_")]:
        df[col] = df[col].fillna(False).astype(bool)


BULL_PATTERNS = ["hammer", "bull_engulf", "morning_star"]
BEAR_PATTERNS = ["shooting_star", "bear_engulf", "evening_star"]


def patterns_at(row, direction):
    names = BULL_PATTERNS if direction == "long" else BEAR_PATTERNS
    return [n for n in names if row[f"p_{n}"]]


# ----------------------------------------------------------------------------- signaux
def market_bias(mkt):
    """+1 haussier / -1 baissier / 0 neutre, selon SPY vs VWAP et EMA21."""
    above = (mkt["close"] > mkt["vwap"]) & (mkt["close"] > mkt["ema21"])
    below = (mkt["close"] < mkt["vwap"]) & (mkt["close"] < mkt["ema21"])
    return pd.Series(np.where(above, 1, np.where(below, -1, 0)), index=mkt.index)


def _stop_level(ref, raw_stop, atr, direction):
    dist = abs(ref - raw_stop)
    dist = min(max(dist, 0.5 * atr), 2.0 * atr)
    return ref - dist if direction == "long" else ref + dist


def find_signals(ticker, df, bias):
    out = []
    rows = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    b = bias.reindex(df.index).fillna(0).values
    used = {}  # (date, strat) -> count / last idx
    n = len(rows)
    for i in range(25, n - 0):
        r = rows.iloc[i]
        if pd.isna(r["atr"]) or r["atr"] <= 0:
            continue
        tod, date = r["tod"], r["date"]
        cands = []

        # ---- ORB
        s = STRATS["ORB"]
        if s["start"] <= tod <= s["end"] and not pd.isna(r["or_high"]):
            pc = rows.iloc[i - 1]["close"]
            if r["close"] > r["or_high"] >= pc and r["rvol"] >= 1.5 and r["close"] > r["vwap"]:
                cands.append(("ORB", "long", _stop_level(r["close"], (r["or_high"] + r["or_low"]) / 2, r["atr"], "long")))
            if r["close"] < r["or_low"] <= pc and r["rvol"] >= 1.5 and r["close"] < r["vwap"]:
                cands.append(("ORB", "short", _stop_level(r["close"], (r["or_high"] + r["or_low"]) / 2, r["atr"], "short")))

        # ---- VWAP pullback
        s = STRATS["VWAP"]
        if s["start"] <= tod <= s["end"] and r["adx"] >= 20:
            prev = rows.iloc[i - 6:i]
            touch_up = max(r["vwap"], r["ema21"]) + 0.2 * r["atr"]
            touch_dn = min(r["vwap"], r["ema21"]) - 0.2 * r["atr"]
            if (r["ema9"] > r["ema21"] and (prev["close"] > prev["vwap"]).sum() >= 5
                    and r["low"] <= touch_up and r["close"] > r["open"] and r["close"] > r["vwap"]):
                cands.append(("VWAP", "long", _stop_level(r["close"], r["low"] - 0.25 * r["atr"], r["atr"], "long")))
            if (r["ema9"] < r["ema21"] and (prev["close"] < prev["vwap"]).sum() >= 5
                    and r["high"] >= touch_dn and r["close"] < r["open"] and r["close"] < r["vwap"]):
                cands.append(("VWAP", "short", _stop_level(r["close"], r["high"] + 0.25 * r["atr"], r["atr"], "short")))

        # ---- Retournement par chandelle
        s = STRATS["REVERS"]
        if s["start"] <= tod <= s["end"] and r["adx"] < 30:
            w = rows.iloc[i - 2:i + 1]
            if patterns_at(r, "long") and w["rsi"].min() <= 30 and (w["low"] <= w["bb_lo"]).any():
                cands.append(("REVERS", "long", _stop_level(r["close"], w["low"].min() - 0.1 * r["atr"], r["atr"], "long")))
            if patterns_at(r, "short") and w["rsi"].max() >= 70 and (w["high"] >= w["bb_up"]).any():
                cands.append(("REVERS", "short", _stop_level(r["close"], w["high"].max() + 0.1 * r["atr"], r["atr"], "short")))

        for strat, direction, stop in cands:
            k = (date, strat)
            cnt, last_i = used.get(k, (0, -99))
            limit = 1 if strat == "ORB" else 2
            if cnt >= limit or i - last_i < 6:
                continue
            used[k] = (cnt + 1, i)
            sign = 1 if direction == "long" else -1
            score, reasons = confluence(r, strat, direction, b[i])
            out.append({
                "ticker": ticker, "strategy": strat, "direction": direction,
                "signal_time": r["ts"], "signal_idx": i, "ref_price": float(r["close"]),
                "stop": float(stop), "atr": float(r["atr"]), "score": score, "reasons": reasons,
                "patterns": ",".join(patterns_at(r, direction)), "rsi": float(r["rsi"]),
                "adx": float(r["adx"]), "rvol": float(r["rvol"]), "market_bias": int(b[i]) * sign,
            })
    return out


def confluence(r, strat, direction, bias):
    sign = 1 if direction == "long" else -1
    reasons, score = [], 0
    if bias * sign > 0:
        score += 1; reasons.append("marché (SPY) dans le même sens")
    if r["rvol"] >= 1.5:
        score += 1; reasons.append(f"volume {r['rvol']:.1f}× la normale")
    if strat in ("ORB", "VWAP") and r["adx"] >= 25:
        score += 1; reasons.append(f"tendance forte (ADX {r['adx']:.0f})")
    if strat == "REVERS" and r["adx"] < 20:
        score += 1; reasons.append(f"marché sans tendance (ADX {r['adx']:.0f})")
    rng = r["high"] - r["low"]
    pos = (r["close"] - r["low"]) / rng if rng > 0 else 0.5
    if patterns_at(r, direction) or (pos >= 0.75 if sign > 0 else pos <= 0.25):
        score += 1; reasons.append("bougie de confirmation")
    if not pd.isna(r["prev_close"]) and (r["close"] - r["prev_close"]) * sign > 0:
        score += 1; reasons.append("du bon côté de la clôture d'hier")
    return score, reasons


def tier(score):
    return "forte" if score >= 4 else ("moyenne" if score >= 2 else "faible")


# ----------------------------------------------------------------------------- simulation
def simulate(sig, df, cfg, now_complete):
    """Retourne le trade simulé (dict) ou None si l'entrée est impossible."""
    i = sig["signal_idx"]
    if i + 1 >= len(df):
        return {**sig, "status": "attente", "entry_time": None}
    nxt = df.iloc[i + 1]
    if nxt["date"] != df.iloc[i]["date"]:
        return None
    sign = 1 if sig["direction"] == "long" else -1
    slip = cfg["slippage_pct"] / 100
    entry = float(nxt["open"]) * (1 + sign * slip)
    stop = sig["stop"]
    risk_ps = (entry - stop) * sign
    if risk_ps <= 0.05 * sig["atr"]:
        return None
    R_t = STRATS[sig["strategy"]]["target_R"]
    target = entry + sign * R_t * risk_ps
    shares = math.floor(cfg["risk_per_trade_usd"] / risk_ps)
    shares = min(shares, math.floor(cfg["account_usd"] * cfg.get("max_leverage", 2) / entry))
    if shares < 1:
        return None
    comm = 2 * max(cfg["commission_min"], cfg["commission_per_share"] * shares)
    risk_usd = risk_ps * shares  # R = gain / risque réellement pris

    exit_px, exit_time, reason = None, None, None
    day = df.iloc[i + 1]["date"]
    j = i + 1
    while j < len(df) and df.iloc[j]["date"] == day:
        bar = df.iloc[j]
        hi, lo, op = bar["high"], bar["low"], bar["open"]
        stop_hit = lo <= stop if sign > 0 else hi >= stop
        tgt_hit = hi >= target if sign > 0 else lo <= target
        if stop_hit:
            gap = (op <= stop) if sign > 0 else (op >= stop)
            px = op if (gap and j > i + 1) else stop
            exit_px, reason = px * (1 - sign * slip), "stop"
        elif tgt_hit:
            exit_px, reason = target, "cible"
        elif bar["tod"] >= LAST_BAR_MIN:
            exit_px, reason = bar["close"] * (1 - sign * slip), "fin de séance"
        if exit_px is not None:
            exit_time = df.index[j]
            break
        j += 1

    t = {**sig, "entry_time": df.index[i + 1], "entry": round(entry, 4), "target": round(target, 4),
         "stop": round(stop, 4), "shares": shares, "target_R": R_t, "risk_usd": round(risk_usd, 2)}
    if exit_px is None:
        last = df.iloc[j - 1]
        if now_complete:  # données manquantes en fin de séance : clôture à la dernière bougie
            exit_px, reason, exit_time = last["close"] * (1 - sign * slip), "fin de séance", df.index[j - 1]
        else:
            mtm = (last["close"] - entry) * sign * shares - comm
            R = mtm / risk_usd
            return {**t, "status": "ouvert", "last_price": float(last["close"]),
                    "R": round(R, 3), "pnl_usd": round(R * cfg["risk_per_trade_usd"], 2)}
    pnl = (exit_px - entry) * sign * shares - comm
    return {**t, "status": "fermé", "exit_time": exit_time, "exit": round(float(exit_px), 4),
            "exit_reason": reason, "R": round(pnl / risk_usd, 3),
            "pnl_usd": round(pnl / risk_usd * cfg["risk_per_trade_usd"], 2)}  # $ normalisés : 100 $ risqués par trade
