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


def season_games(S, live_from):
    """Every completed game this season: the model's spread pick graded at the posted closing line, and the info
    the page needs to grade wind unders with archived near-kickoff forecasts (outdoor, non-retractable venues)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from .features import venue_for, RETRACTABLE_HOME
    ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
    d, _, _ = pickle.load(open(os.path.join(data.DATA_DIR, "v3_signals.pkl"), "rb"))
    adj = V.oos_adjust(d, ["st_diff"], [S])
    d = d.assign(adj=adj)
    st = d[(d.season == S)].set_index("game_id")
    g = data.load_games()
    g = g[(g.season == S) & g.result.notna()]
    out = []
    for r in g.sort_values(["week", "gameday", "gametime"]).itertuples():
        a = st.adj.get(r.game_id) if r.game_id in st.index else None
        a = None if a is None or a != a else round(float(a), 2)
        hh, mm = map(int, str(r.gametime).split(":")) if isinstance(r.gametime, str) else (13, 0)
        ko = datetime(r.gameday.year, r.gameday.month, r.gameday.day, hh, mm, tzinfo=ET).astimezone(UTC)
        v = venue_for(r.home_team, r.location, r.stadium, r.roof)
        outdoor = bool(v and v["outdoor"] and r.home_team not in RETRACTABLE_HOME and r.roof == "outdoors")
        rec = {"id": r.game_id, "wk": int(r.week), "h": r.home_team, "a": r.away_team, "hs": int(r.home_score), "as": int(r.away_score),
               "L": None if r.spread_line != r.spread_line else float(r.spread_line), "adj": a,
               "oh": None if r.home_spread_odds != r.home_spread_odds else int(r.home_spread_odds),
               "oa": None if r.away_spread_odds != r.away_spread_odds else int(r.away_spread_odds),
               "T": None if r.total_line != r.total_line else float(r.total_line),
               "ou": None if r.under_odds != r.under_odds else int(r.under_odds),
               "live": r.gameday.strftime("%Y-%m-%d") >= live_from, "ko": ko.strftime("%Y-%m-%dT%H:%M:%SZ")}
        if outdoor:
            rec["lat"], rec["lon"] = round(v["lat"], 3), round(v["lon"], 3)
        out.append(rec)
    return out


def enrich(path):
    wk = json.load(open(path))
    S = int(wk["season"])
    from .version import FROZEN
    h = {"season_to_date": season_record(S), "injury_report_week": inj_asof(S), "season_games": season_games(S, FROZEN),
         "live_from": FROZEN}
    wk["health"] = h
    json.dump(wk, open(path, "w"), separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return h
