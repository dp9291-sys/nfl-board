"""Four pre-declared edge tests (python -m nflats.edges LINE_HISTORY.json).

Primary rules (fixed before running):
 1 wind unders      : outdoor/open roof, game wind >= 10 mph -> under the closing total at the posted price
 2 teaser legs      : 6-pt teaser legs on favourites of 7.5-8.5 (to 1.5-2.5) and dogs of 1.5-2.5 (to 7.5-8.5);
                      success = leg win rate above 73.9% (2-team at -120); also shown vs -130/-135/-140
 3 early timing     : special-teams bets (lean >= 0.5) where the full stats line is >= 1.5 pts off the opener
                      on the same side -> bet at the opener vs near kickoff (best Arizona book)
 4 Pinnacle gap     : Arizona book's opener >= 1.5 pts off Pinnacle's opener -> take the extra points there
Seasons before 2015 were never used in development, so they are an independent check for tests 1 and 2.
"""
import json
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from . import data, validate as V, linehistory as LH

PAY = LH.PAY
AZ = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet"]


def rec(win, push=None, odds=None, be=0.5238):
    win = pd.Series(np.asarray(win)).astype(bool)
    push = pd.Series(np.zeros(len(win), bool) if push is None else np.asarray(push)).astype(bool)
    dec = win[~push.values]
    n = len(dec)
    if n == 0:
        return {"n": 0}
    out = {"n": int(n), "win": round(100 * float(dec.mean()), 1),
           "p": round(float(binomtest(int(dec.sum()), n, be, alternative="greater").pvalue), 4)}
    if odds is not None:
        o = pd.Series(np.asarray(odds, float)).fillna(-110)
        pnl = np.where(push.values, 0.0, np.where(win.values, [PAY(x) for x in o], -1.0))
        out["roi"] = round(100 * float(pnl.mean()), 1)
    return out


def test_wind(g):
    g = g[g.total.notna() & g.total_line.notna() & g.wind.notna() & g.roof.isin(["outdoors", "open"])]
    out = {}
    for lab, lo, hi in [("1999-2014 (independent)", 1999, 2014), ("2015-2026", 2015, 2026)]:
        for th in (10, 15):
            x = g[g.season.between(lo, hi) & (g.wind >= th)]
            out[f"wind {th}+ mph, {lab}"] = rec(x.total < x.total_line, x.total == x.total_line, x.under_odds)
        x = g[g.season.between(lo, hi) & (g.wind < 10)]
        out[f"wind under 10 (control), {lab}"] = rec(x.total < x.total_line, x.total == x.total_line, x.under_odds)
    x = g[g.season >= 2015]
    out["by era (10+ mph)"] = {f"{a}-{b}": rec((y := x[x.season.between(a, b) & (x.wind >= 10)]).total < y.total_line,
                                               y.total == y.total_line, y.under_odds)["win"]
                               for a, b in [(2015, 2018), (2019, 2022), (2023, 2026)]}
    return out


def test_teasers(g):
    g = g[g.result.notna() & g.spread_line.notna()].copy()
    legs = []
    for r in g.itertuples():
        L = r.spread_line  # expected home margin
        for side, fav_by in (("home", L), ("away", -L)):
            margin = r.result if side == "home" else -r.result   # side's margin
            if 7.5 <= fav_by <= 8.5:
                legs.append((r.season, "favourite -7.5 to -8.5", margin - (fav_by - 6) > 0, r.total_line))
            elif -2.5 <= fav_by <= -1.5:
                legs.append((r.season, "underdog +1.5 to +2.5", margin + (-fav_by + 6) > 0, r.total_line))
    T = pd.DataFrame(legs, columns=["season", "kind", "win", "total"])
    out = {}
    for lab, lo, hi in [("1999-2014 (independent)", 1999, 2014), ("2015-2026", 2015, 2026)]:
        x = T[T.season.between(lo, hi)]
        out[f"all Wong legs, {lab}"] = rec(x.win, be=0.739)
        for k, y in x.groupby("kind"):
            out[f"  {k}, {lab}"] = rec(y.win, be=0.739)
        y = x[x.total <= 49]
        out[f"  total 49 or less, {lab}"] = rec(y.win, be=0.739)
    out["break-even per leg (2-team)"] = {"-110": 72.4, "-120": 73.9, "-130": 75.2, "-135": 75.8, "-140": 76.4}
    x = T[T.season >= 2015]
    out["by era (all legs)"] = {f"{a}-{b}": round(100 * float(x[x.season.between(a, b)].win.mean()), 1)
                                for a, b in [(2015, 2018), (2019, 2022), (2023, 2026)]}
    return out


def test_timing(raw, d):
    G = LH.per_game(raw)
    X = d.merge(G, on="game_id")
    X = X[X.result.notna() & X.L_open.notna() & X.L_close.notna() & (X.st_adj.abs() >= 0.5)]
    out = {}
    hb = {s: LH.best_price(raw[raw.book.isin(AZ)], s) for s in ("open", "close")}
    side = np.sign(X.st_adj)
    stats_line = X.spread_line + X.stats_gap
    gap = stats_line - X.L_open
    agree = (np.sign(gap) == side) & (gap.abs() >= 1.5)
    for lab, m in [("stats model agrees (1.5+ off opener)", agree), ("does not agree", ~agree)]:
        for st in ("open", "close"):
            h, a = hb[st]
            Y = X[m].merge(h, left_on="game_id", right_index=True).merge(a, left_on="game_id", right_index=True)
            hm = Y.st_adj > 0
            L = np.where(hm, Y.h_L, Y.a_L)
            o = np.where(hm, Y.h_oh, Y.a_oa)
            res = np.where(hm, Y.result - L, L - Y.result)
            out[f"{lab}: bet at {st}"] = rec(res > 0, res == 0, o)
        mv = (side * (X.L_close - X.L_open))[m]
        out[f"{lab}: line moved toward bet / against (%)"] = [round(100 * float((mv > 0).mean()), 1), round(100 * float((mv < 0).mean()), 1)]
    return out


def test_pinnacle(raw, results):
    op = raw[raw.kind == "open"]
    pin = op[op.book == "pinnacle"].groupby("game_id").L.median().rename("pinL")
    cl = raw[raw.kind == "close"]
    pin_c = cl[cl.book == "pinnacle"].groupby("game_id").L.median().rename("pin_close")
    b = op[op.book.isin(AZ) & op.L.notna()].join(pin, on="game_id").join(pin_c, on="game_id").merge(results, on="game_id")
    b = b[b.pinL.notna() & b.result.notna()]
    b["gap"] = b.L - b.pinL
    b["side"] = np.where(b.gap > 0, -1, 1)
    b["odds"] = np.where(b.side > 0, b.oh, b.oa)
    b["absgap"] = b.gap.abs()
    out = {}
    for th in (1.0, 1.5, 2.0):
        c = b[b.absgap >= th].sort_values(["game_id", "absgap", "odds"], ascending=[True, False, False]).groupby("game_id").first()
        res = (c.result - c.L) * c.side
        k = f"gap {th}+ pts" + (" (primary)" if th == 1.5 else "")
        out[k] = rec(res > 0, res == 0, c.odds)
        if th == 1.5:
            out["  by season"] = {int(s): rec(((y.result - y.L) * y.side) > 0, ((y.result - y.L) * y.side) == 0)["win"] for s, y in c.groupby("season")}
            out["  Pinnacle close vs that book's opener: moved toward bet %"] = round(100 * float((((c.pin_close - c.L) * -c.side) < 0).mean()), 1)
            out["  by book"] = {bk: int(n) for bk, n in c.book.value_counts().items()}
    return out


def main(path):
    g = data.load_games()
    g = g[g.game_type == "REG"]
    raw = LH.load(path)
    d, _, _ = pickle.load(open(f"{data.DATA_DIR}/v3_signals.pkl", "rb"))
    tests = [s for s in sorted(d.season.unique()) if s >= 2016]
    d = d.assign(st_adj=V.oos_adjust(d, ["st_diff"], tests))
    res = {"1 wind unders": test_wind(g), "2 teaser legs": test_teasers(g),
           "3 early timing": test_timing(raw, d), "4 Pinnacle gap at opener": test_pinnacle(raw, d[["game_id", "season", "result"]])}
    prim = {"1": res["1 wind unders"]["wind 10+ mph, 2015-2026"]["p"],
            "1 holdout": res["1 wind unders"]["wind 10+ mph, 1999-2014 (independent)"]["p"],
            "2": res["2 teaser legs"]["all Wong legs, 2015-2026"]["p"],
            "2 holdout": res["2 teaser legs"]["all Wong legs, 1999-2014 (independent)"]["p"],
            "3": res["3 early timing"]["stats model agrees (1.5+ off opener): bet at open"]["p"],
            "4": res["4 Pinnacle gap at opener"]["gap 1.5+ pts (primary)"]["p"]}
    main_p = [prim["1"], prim["2"], prim["3"], prim["4"]]
    res["FDR (BH q=0.10) on the four primary tests"] = dict(zip(["1", "2", "3", "4"], V.bh(main_p, q=0.10)))
    res["primary p-values"] = prim
    return res


if __name__ == "__main__":
    out = main(sys.argv[1])
    for k, v in out.items():
        print("==", k)
        if isinstance(v, dict):
            for kk, vv in v.items():
                print("  ", kk, json.dumps(vv, default=str))
        else:
            print("  ", v)
