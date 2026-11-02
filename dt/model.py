"""
Modèle de probabilité et statistiques.

Pour chaque signal, on estime sans regarder le futur (seulement les trades fermés AVANT le signal) :
- p_win : probabilité que le trade finisse gagnant (loi Bêta, prior sceptique = seuil de rentabilité)
- ev_R  : gain moyen attendu en R (moyenne bayésienne « rétrécie » vers 0)
Hiérarchie : stratégie -> (stratégie, sens, force de la confluence).
Le portefeuille « filtré » ne prend que les signaux avec ev_R >= seuil (comme les paris +EV).
"""
import math
from collections import defaultdict

import numpy as np

from .engine import STRATS, tier

K_STRAT_P, K_BUCKET_P = 6.0, 10.0
K_STRAT_EV, K_BUCKET_EV = 15.0, 10.0


class Bayes:
    def __init__(self):
        self.stats = defaultdict(lambda: [0, 0, 0.0])  # n, wins, sumR

    def add(self, key, R):
        s = self.stats[key]
        s[0] += 1
        s[1] += 1 if R > 0 else 0
        s[2] += R

    def estimate(self, strat, direction, score):
        R_t = STRATS[strat]["target_R"]
        p0 = 1 / (1 + R_t)  # seuil de rentabilité : prior sceptique
        n_s, w_s, sum_s = self.stats[(strat,)]
        p_s = (w_s + K_STRAT_P * p0) / (n_s + K_STRAT_P)
        ev_s = sum_s / (n_s + K_STRAT_EV)  # rétréci vers 0
        bkey = (strat, direction, tier(score))
        n_b, w_b, sum_b = self.stats[bkey]
        p_b = (w_b + K_BUCKET_P * p_s) / (n_b + K_BUCKET_P)
        ev_b = (sum_b + K_BUCKET_EV * ev_s) / (n_b + K_BUCKET_EV)
        return {"p_win": round(p_b, 4), "ev_R": round(ev_b, 4), "n_strat": n_s, "n_bucket": n_b}


def score_trades(trades, cfg):
    """Ajoute p_win / ev_R / pris_filtre à chaque trade, dans l'ordre chronologique, sans regarder le futur."""
    trades.sort(key=lambda t: t["signal_time"])
    bayes = Bayes()
    closed_queue = sorted([t for t in trades if t.get("status") == "fermé"], key=lambda t: t["exit_time"])
    qi = 0
    open_filtered = []
    day_R = defaultdict(float)
    for t in trades:
        st = t["signal_time"]
        while qi < len(closed_queue) and closed_queue[qi]["exit_time"] < st:
            c = closed_queue[qi]
            bayes.add((c["strategy"],), c["R"])
            bayes.add((c["strategy"], c["direction"], tier(c["score"])), c["R"])
            qi += 1
        t.update(bayes.estimate(t["strategy"], t["direction"], t["score"]))
        t["tier"] = tier(t["score"])

        # Portefeuille filtré (+EV), avec limites de risque
        take = (t["ev_R"] >= cfg["min_ev_R"] and t["n_strat"] >= cfg["min_samples_for_ev"]
                and t.get("status") in ("fermé", "ouvert", "attente"))
        reason = "" if take else ("EV insuffisante" if t["ev_R"] < cfg["min_ev_R"] else "pas assez d'historique")
        day = st.date()
        if take:
            open_filtered = [o for o in open_filtered if o["signal_time"].date() == day
                             and (o.get("exit_time") is None or o["exit_time"] > st)]
            if len(open_filtered) >= cfg["max_open_filtered"]:
                take, reason = False, "trop de positions ouvertes"
            elif day_R[day] <= -cfg["daily_loss_limit_R"]:
                take, reason = False, "limite de perte du jour atteinte"
        t["pris_filtre"] = bool(take)
        t["raison_filtre"] = reason
        if take:
            open_filtered.append(t)
            if t.get("status") == "fermé":
                day_R[day] += t["R"]
    return trades


# ----------------------------------------------------------------------------- statistiques
def summarize(trades, cfg):
    closed = [t for t in trades if t.get("status") == "fermé"]
    n = len(closed)
    if n == 0:
        return {"n": 0}
    R = np.array([t["R"] for t in closed])
    wins = R[R > 0]
    losses = R[R <= 0]
    eq = np.cumsum(R)
    peak = np.maximum.accumulate(np.concatenate([[0], eq]))[1:]
    dd = float((eq - peak).min()) if n else 0.0
    mean = float(R.mean())
    sd = float(R.std(ddof=1)) if n > 1 else float("nan")
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    days = sorted({t["signal_time"].date() for t in closed})
    return {
        "n": n, "win_rate": round(float((R > 0).mean()), 4), "avg_R": round(mean, 4),
        "sum_R": round(float(R.sum()), 2), "pnl_usd": round(float(sum(t["pnl_usd"] for t in closed)), 2),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else None,
        "max_dd_R": round(dd, 2), "ci_low": round(mean - half, 4) if n > 1 else None,
        "ci_high": round(mean + half, 4) if n > 1 else None, "avg_win_R": round(float(wins.mean()), 3) if len(wins) else None,
        "avg_loss_R": round(float(losses.mean()), 3) if len(losses) else None, "days": len(days),
        "pct_cible": round(sum(t["exit_reason"] == "cible" for t in closed) / n, 3),
        "pct_stop": round(sum(t["exit_reason"] == "stop" for t in closed) / n, 3),
    }


def equity_curve(trades, cfg):
    closed = sorted([t for t in trades if t.get("status") == "fermé"], key=lambda t: t["exit_time"])
    by_day = defaultdict(float)
    for t in closed:
        by_day[t["exit_time"].date().isoformat()] += t["pnl_usd"]
    out, eq = [], cfg["account_usd"]
    for d in sorted(by_day):
        eq += by_day[d]
        out.append({"date": d, "equity": round(eq, 2), "jour": round(by_day[d], 2)})
    return out


def calibration(trades):
    closed = [t for t in trades if t.get("status") == "fermé"]
    bins = [(0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)]
    out = []
    for lo, hi in bins:
        sel = [t for t in closed if lo <= t["p_win"] < hi]
        if sel:
            out.append({"bin": f"{int(lo*100)}–{min(int(hi*100),100)} %", "n": len(sel),
                        "prevu": round(float(np.mean([t["p_win"] for t in sel])), 3),
                        "reel": round(float(np.mean([t["R"] > 0 for t in sel])), 3)})
    return out


def breakdown(trades, cfg, key):
    groups = defaultdict(list)
    for t in trades:
        groups[key(t)].append(t)
    rows = []
    for k, ts in groups.items():
        s = summarize(ts, cfg)
        if s.get("n"):
            rows.append({"cle": k, **s})
    return sorted(rows, key=lambda r: -r["n"])
