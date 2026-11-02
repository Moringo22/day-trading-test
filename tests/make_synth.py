"""Génère des bougies 5 min SIMULÉES (marche aléatoire) pour tester le code sans Internet.
python tests/make_synth.py <dossier_racine> <date_fin AAAA-MM-JJ> [heure_fin HH:MM]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
end = pd.Timestamp(sys.argv[2])
end_hm = sys.argv[3] if len(sys.argv) > 3 else "16:00"
cfg = json.loads((root / "config.json").read_text())
rng = np.random.default_rng(42)
days = pd.bdate_range(end - pd.Timedelta(days=84), end)
mkt = rng.normal(0, 0.0012, (len(days), 78))
(root / "data" / "bars").mkdir(parents=True, exist_ok=True)
allrows = []
for k, t in enumerate(cfg["tickers"]):
    px = 50 + 40 * k
    rows = []
    for di, d in enumerate(days):
        px *= 1 + rng.normal(0, 0.008)
        for b in range(78):
            ts = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(minutes=570 + 5 * b)
            if d == days[-1] and ts.strftime("%H:%M") >= end_hm:
                break
            ret = 0.8 * mkt[di, b] + rng.normal(0, 0.0015) + (0.0004 * np.sin(b / 6) if k % 2 else 0)
            o = px
            c = px * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.0007)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.0007)))
            v = int(1e5 * (2.5 - 1.5 * np.sin(np.pi * b / 77)) * rng.lognormal(0, 0.4))
            rows.append([ts.isoformat(), t, o, h, l, c, v])
            px = c
    allrows.extend(rows)
df = pd.DataFrame(allrows, columns=["ts", "ticker", "open", "high", "low", "close", "volume"])
df["day"] = df["ts"].str[:10]
for d, part in df.groupby("day"):
    part.drop(columns="day").to_csv(root / "data" / "bars" / f"{d}.csv", index=False)
print("ok", len(days), "jours")
