"""Walk-forward tests of the luck-stripped special-teams variants (python -m nflats.stluck_test).

Bets follow the board's rule: lean = walk-forward weight x rating difference, bet when |lean| >= 0.5,
graded at the closing spread, pushes dropped. Weights are learned only from earlier seasons (2000 on),
test seasons 2003-2026. Pre-registered questions, all under one Benjamini-Hochberg FDR (q = 0.10):
  1. Does each variant add anything on top of the current rating? (residual walk-forward t-test)
  2. Indicator: among the current rule's bets, do games where the variant agrees in direction
     (variant's own lean >= 0.25 the same way) win more often than games where it doesn't?
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from . import data, validate as V

TESTS = list(range(2003, 2027))
VARS = ["all", "A", "AF", "B", "BF", "C", "CF"]
NAMES = {"all": "current + field goals as they happened", "A": "kicking luck out (XPs at expected), big plays capped",
         "AF": "A + field goals at expected value", "B": "A + punts weather/venue-adjusted",
         "BF": "B + field goals at expected value", "C": "B, opponent-adjusted", "CF": "C + field goals at expected value"}


def load():
    g = data.load_games()
    g = g[(g.season >= 2000) & g.result.notna() & g.spread_line.notna()]
    r = pd.read_parquet(os.path.join(data.DATA_DIR, "stluck.parquet"))
    d = g[["game_id", "season", "week", "home_team", "away_team", "spread_line", "result"]].merge(r, on="game_id")
    d["ats"] = (d.result - d.spread_line).clip(-21, 21)
    d["ats_raw"] = d.result - d.spread_line
    return d.reset_index(drop=True)


def bets(d, adj, thr=0.5, mask=None):
    m = d.season.isin(TESTS) & adj.notna() & (adj.abs() >= thr) & (d.ats_raw != 0)
    if mask is not None:
        m &= mask
    return m, (np.sign(adj) == np.sign(d.ats_raw))


def rate(m, win, d, seasons=None):
    mm = m if seasons is None else m & d.season.isin(seasons)
    n = int(mm.sum())
    return {"n": n, "win": round(100 * float(win[mm].mean()), 1) if n else None}


def clustered_diff(d, m1, m2, win, B=4000, seed=11):
    """Season-clustered bootstrap of win-rate(m1) - win-rate(m2)."""
    rng = np.random.default_rng(seed)
    S = np.array(TESTS)
    g1 = {s: win[m1 & (d.season == s)].values for s in S}
    g2 = {s: win[m2 & (d.season == s)].values for s in S}
    obs = np.concatenate(list(g1.values())).mean() - np.concatenate(list(g2.values())).mean()
    bs = []
    for _ in range(B):
        pick = rng.choice(S, len(S))
        a = np.concatenate([g1[s] for s in pick]); b = np.concatenate([g2[s] for s in pick])
        if len(a) and len(b):
            bs.append(a.mean() - b.mean())
    bs = np.array(bs)
    p = float((bs <= 0).mean())  # one-sided: agreement wins more
    return round(100 * obs, 1), [round(100 * x, 1) for x in np.percentile(bs, [5, 95])], p


def run():
    d = load()
    out, pv = {}, {}
    base = V.oos_adjust(d, ["sl_raw"], TESTS, first=2000)
    m0, w0 = bets(d, base)
    out["current rule"] = {"2003-26": rate(m0, w0, d), "2013-26": rate(m0, w0, d, range(2013, 2027)), "2026": rate(m0, w0, d, [2026]),
                           "t": V.score(d[d.season.isin(TESTS)], base[d.season.isin(TESTS)])["t"]}
    d2 = d.copy()
    d2["ats"] = (d.ats - base.fillna(0)).clip(-21, 21)
    for k in VARS:
        col = f"sl_{k}"
        adj = V.oos_adjust(d, [col], TESTS, first=2000)
        sc = V.score(d[d.season.isin(TESTS)], adj[d.season.isin(TESTS)])
        m, w = bets(d, adj)
        inc = V.score(d2[d2.season.isin(TESTS)], V.oos_adjust(d2, [col], TESTS, first=2000)[d2.season.isin(TESTS)])
        # indicator: among current-rule bets, does the variant agree?
        agree = m0 & (np.sign(adj) == np.sign(base)) & (adj.abs() >= 0.25)
        dis = m0 & ~agree
        diff, ci, p_ag = clustered_diff(d, agree, dis, w0)
        out[k] = {"name": NAMES[k], "alone_t": sc["t"], "as_rule_2003-26": rate(m, w, d), "as_rule_2013-26": rate(m, w, d, range(2013, 2027)),
                  "as_rule_2026": rate(m, w, d, [2026]), "on_top_t": inc["t"], "on_top_p": round(inc["p"], 4),
                  "agree": rate(agree, w0, d), "disagree": rate(dis, w0, d),
                  "agree_2026": rate(agree, w0, d, [2026]), "disagree_2026": rate(dis, w0, d, [2026]),
                  "agree_minus_disagree": diff, "ci90": ci, "agree_p": round(p_ag, 4)}
        pv[f"on top: {k}"] = inc["p"]
        pv[f"indicator: {k}"] = p_ag
    keys = list(pv)
    ok = V.bh([pv[k] for k in keys], q=0.10)
    out["FDR passes (BH q=0.10)"] = [k for k, o in zip(keys, ok) if o]
    return out


if __name__ == "__main__":
    res = run()
    json.dump(res, open(os.path.join(data.DATA_DIR, "stluck_results.json"), "w"), indent=1)
    for k, v in res.items():
        print(k, json.dumps(v))
