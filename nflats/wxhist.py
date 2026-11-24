"""Wind-under validation with archived forecasts (python -m nflats.wxhist WEATHER.json [LINE_HISTORY.json ...]).

Rules fixed before the data arrived:
 primary  : under at the closing total (posted price) when the archived near-kickoff forecast at the kickoff hour
            is 10+ mph sustained, outdoor games 2021-2026; shown beside the actual-wind rule on the same games
 secondary: 1/2/3-day-out forecasts (2024+); forecast bands 10-14 / 15-19 / 20+; forecast gusts 20+;
            market reaction: total move open->close vs the 2-day-out forecast, and under at the opener vs the close
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from . import data, linehistory as LH

PAY = lambda o: o / 100 if o > 0 else 100 / abs(o)


def rec(res, odds=None):
    res = np.asarray(res, float)
    dec = res[res != 0]
    if not len(dec):
        return {"n": 0}
    w = int((dec > 0).sum())
    o = {"n": int(len(dec)), "win": round(100 * w / len(dec), 1),
         "p_vs_52.4": round(float(binomtest(w, len(dec), 0.524, alternative="greater").pvalue), 3)}
    if odds is not None:
        od = pd.Series(np.asarray(odds, float)).fillna(-110).values
        pnl = np.where(res > 0, [PAY(x) for x in od], np.where(res < 0, -1.0, 0.0))
        o["roi"] = round(100 * float(pnl.mean()), 1)
    return o


def frame(path):
    j = json.load(open(path))["games"]
    rows = []
    for gid, r in j.items():
        h, p = r.get("h") or {}, r.get("p") or {}
        rows.append({"game_id": gid, "f0": h.get("wind_speed_10m"), "f0_avg": h.get("wind_speed_10m_avg"), "g0": h.get("wind_gusts_10m"),
                     "f1": p.get("wind_speed_10m_previous_day1"), "f2": p.get("wind_speed_10m_previous_day2"),
                     "f3": p.get("wind_speed_10m_previous_day3"), "h_err": h.get("err"), "p_err": p.get("err")})
    f = pd.DataFrame(rows)
    g = data.load_games()
    g = g[g.total.notna() & g.total_line.notna()]
    x = f.merge(g[["game_id", "season", "week", "total", "total_line", "under_odds", "wind"]], on="game_id", how="inner")
    # actual wind: schedule file, else play-by-play game weather
    miss = x.wind.isna()
    if miss.any():
        for S in sorted(x.season[miss].unique()):
            p = os.path.join(data.DATA_DIR, f"pbp_{S}.parquet")
            if os.path.exists(p):
                w = pd.read_parquet(p, columns=["game_id", "wind"]).groupby("game_id").wind.first()
                x.loc[x.season == S, "wind"] = x.loc[x.season == S, "wind"].fillna(x.loc[x.season == S, "game_id"].map(w))
    x["res_under"] = x.total_line - x.total
    return x


def main(paths):
    x = frame(paths[0])
    R = {"games with archived forecast": int(x.f0.notna().sum()), "with 1-3 day forecasts": int(x.f2.notna().sum()),
         "errors": {"near-kickoff": int(x.h_err.notna().sum()), "earlier-day": int(x.p_err.notna().sum())},
         "forecast vs actual wind (mean abs diff, corr)": [round(float((x.f0 - x.wind).abs().mean()), 2), round(float(x.f0.corr(x.wind)), 3)]}
    both = x[x.f0.notna() & x.wind.notna()]
    R["PRIMARY: under, forecast 10+ (near kickoff)"] = rec(both.res_under[both.f0 >= 10], both.under_odds[both.f0 >= 10])
    R["  same games, actual wind 10+"] = rec(both.res_under[both.wind >= 10], both.under_odds[both.wind >= 10])
    R["  forecast 10+ but actual under 10"] = rec(both.res_under[(both.f0 >= 10) & (both.wind < 10)])
    R["  actual 10+ but forecast under 10"] = rec(both.res_under[(both.f0 < 10) & (both.wind >= 10)])
    R["  control: forecast under 10"] = rec(both.res_under[both.f0 < 10])
    for lo, hi in [(10, 15), (15, 20), (20, 99)]:
        m = (x.f0 >= lo) & (x.f0 < hi)
        R[f"  forecast {lo}-{hi if hi < 99 else '+'} mph"] = rec(x.res_under[m], x.under_odds[m])
    R["  forecast gusts 20+"] = rec(x.res_under[x.g0 >= 20], x.under_odds[x.g0 >= 20])
    R["  kickoff-to-+3h average forecast 10+"] = rec(x.res_under[x.f0_avg >= 10], x.under_odds[x.f0_avg >= 10])
    for k, lab in (("f1", "1 day out"), ("f2", "2 days out"), ("f3", "3 days out")):
        m = x[k] >= 10
        R[f"forecast {lab} 10+ (2024+)"] = rec(x.res_under[m], x.under_odds[m])
    if len(paths) > 1:
        tmp = "/tmp/_lh_merged.json"
        snaps = {}
        for p in paths[1:]:
            snaps.update(json.load(open(p))["snaps"])
        json.dump({"snaps": snaps}, open(tmp, "w"))
        G = LH.per_game(LH.load(tmp))
        y = x.merge(G[["game_id"] + [c for c in G.columns if c.startswith("tot_")]], on="game_id", how="left")
        if "tot_open" in y and "tot_close" in y:
            mv = (y.tot_close - y.tot_open)
            for k, lab in (("f2", "2 days out"), ("f0", "near kickoff")):
                m = y[k].notna() & mv.notna()
                if m.sum() > 30:
                    sl = np.polyfit(y.loc[m, k], mv[m], 1)[0]
                    R[f"total move open->close per +1 mph ({lab} forecast)"] = round(float(sl), 3)
                    w = m & (y[k] >= 10)
                    R[f"  avg total move when {lab} forecast 10+"] = round(float(mv[w].mean()), 2) if w.any() else None
            m = y.f2.notna() & y.tot_open.notna() & (y.f2 >= 10)
            R["under at the OPENING total when 2-day forecast 10+"] = rec((y.tot_open - y.total)[m])
            R["  same games at the closing total"] = rec((y.tot_close - y.total)[m])
    return R


if __name__ == "__main__":
    for k, v in main(sys.argv[1:]).items():
        print(k, json.dumps(v))
