"""Model-health facts added to the week file each build (season-to-date record of both bet types, data freshness)."""
import json
import os
import pickle
import numpy as np
import pandas as pd
from . import data, validate as V


def _rec(res):
    res = np.asarray(res, float)
    return {"w": int((res > 0).sum()), "l": int((res < 0).sum()), "p": int((res == 0).sum())}


def season_record(S):
    d, _, _ = pickle.load(open(os.path.join(data.DATA_DIR, "v3_signals.pkl"), "rb"))
    adj = V.oos_adjust(d, ["st_diff"], [S])
    x = d.assign(adj=adj)
    x = x[(x.season == S) & x.result.notna() & x.spread_line.notna() & (x.adj.abs() >= 0.5)]
    res = np.where(x.adj > 0, x.result - x.spread_line, x.spread_line - x.result)
    weeks = {int(w): _rec(res[(x.week == w).values]) for w in sorted(x.week.unique())}
    out = {"spread": {"total": _rec(res), "by_week": weeks, "basis": "special-teams lean 0.5+ at the posted closing line"}}
    # wind unders: outdoor games with 10+ mph game wind (games file, else play-by-play weather)
    g = data.load_games()
    g = g[(g.season == S) & g.total.notna() & g.total_line.notna() & g.roof.isin(["outdoors", "open"])].copy()
    p = os.path.join(data.DATA_DIR, f"pbp_{S}.parquet")
    if os.path.exists(p) and len(g):
        w = pd.read_parquet(p, columns=["game_id", "wind"]).groupby("game_id").wind.first()
        g["wind"] = g.wind.fillna(g.game_id.map(w))
    u = g[g.wind >= 10]
    out["under"] = {"total": _rec((u.total_line - u.total).values), "n_games_with_wind_data": int(g.wind.notna().sum()),
                    "basis": "under the closing total in outdoor games with 10+ mph game-time wind"}
    return out


def inj_asof(S):
    """Latest week with a published injury report (the file has no timestamps)."""
    p = os.path.join(data.DATA_DIR, f"inj_{S}.parquet")
    if not os.path.exists(p):
        return None
    x = pd.read_parquet(p, columns=["week"])
    return int(x.week.max()) if len(x) else None


def enrich(path):
    wk = json.load(open(path))
    S = int(wk["season"])
    h = {"season_to_date": season_record(S), "injury_report_week": inj_asof(S)}
    wk["health"] = h
    json.dump(wk, open(path, "w"), separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return h
