"""Rigorous ATS validation.

1. Each candidate signal is turned into an ATS adjustment by a ridge fit on PAST seasons only
   (target = result - closing spread), then scored on the next season. Test seasons 2015-2026.
2. Scores: out-of-sample correlation with ATS margin (t-stat, one-sided p), hit rate on the third of
   games where the signal is strongest, stability across three eras.
3. Benjamini-Hochberg false-discovery control across all candidates.
4. Final model with NESTED selection: for test season S, signals are chosen using only their
   out-of-sample record in seasons before S, then refit on seasons before S and applied to S.
5. Bootstrap (by week) confidence intervals, ROI at -110 and -105.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest
from .model import ridge_fit, ridge_pred, season_w
from . import model2 as M2

CANDIDATES = {
    # special teams
    "st_diff": "Special teams, net EPA/game", "st_fg_diff": "Special teams: field goals & XPs",
    "st_punt_diff": "Special teams: punting", "st_ko_diff": "Special teams: kickoffs",
    # stats vs market
    "stats_gap": "Full 34-factor stats line vs close", "model_line_gap_pts": "Points-based scoring model vs close",
    # QB / injuries
    "qb_adj_diff": "QB starter vs team baseline (EPA)", "qb_cpoe_diff": "QB starter vs baseline (CPOE)",
    "inj_off_diff": "Injuries, offense (snap-weighted)", "inj_def_diff": "Injuries, defense (snap-weighted)",
    "cl_ol_diff": "2+ O-line starters out", "cl_db_diff": "2+ secondary starters out", "cl_rec_diff": "2+ receivers out",
    # matchups / style / coaching
    "pass_match": "Pass offense x opposing pass defense", "sack_match": "Sack-prone line x pass rush",
    "go_rate_diff": "4th-down aggressiveness", "new_hc_early_diff": "New head coach, weeks 1-6",
    "proe_diff": "Pass rate over expected",
    # luck / regression
    "to_diff": "Interception rates (regression)", "fum_diff": "Fumble rates (regression)",
    "luck_diff": "Scoreboard vs efficiency (luck)", "last_ats_diff": "Last game's cover margin",
    # schedule
    "rest_diff": "Rest days", "early_west": "West-coast team, early kickoff", "tz_shift": "Time-zone shift",
    "div": "Division game",
    # totals & market
    "total_dev_x_fav": "Market total vs stats total (with favourite)", "fav_low_total": "Big spread relative to total",
    "wind_x_fav": "Wind 10+ mph (with favourite)", "mkt_dev": "Line move vs market's own ratings",
}
ERAS = [(2015, 2018), (2019, 2022), (2023, 2026)]


def prep(f3):
    d = M2.prep(f3).copy()
    d["ats"] = (d.result - d.spread_line).clip(-21, 21)
    d["ats_raw"] = d.result - d.spread_line
    feats = list(__import__("nflats.model", fromlist=["POWER_FEATS"]).POWER_FEATS) + sum(M2.GROUPS.values(), [])
    proj = M2.project(d, feats)
    d["stats_gap"] = proj - d.spread_line
    for c in CANDIDATES:
        d[c] = d[c].astype(float).fillna(0.0)
    return d


def oos_adjust(d, cols, test_seasons, first=2013, lam=50.0):
    out = pd.Series(np.nan, index=d.index)
    for s in test_seasons:
        tr = d[(d.season >= first) & (d.season < s)]
        te = d.season == s
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        b = ridge_fit(((tr[cols] - mu) / sd).values, tr.ats.values, lam=lam, w=season_w(tr.season, s))
        b[0] = 0.0  # no generic home/away bias term
        out[te] = ridge_pred(b, ((d.loc[te, cols] - mu) / sd).values)
    return out


def score(d, adj, mask=None):
    m = adj.notna() & (d.ats_raw != 0)
    if mask is not None:
        m &= mask
    x, y = adj[m], d.ats[m]
    n = int(m.sum())
    r = float(np.corrcoef(x, y)[0, 1]) if n > 10 and x.std() > 0 else 0.0
    t = r * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - r * r, 1e-9))
    # strongest third of games per season
    thr = adj[m].abs().groupby(d.season[m]).transform(lambda s: s.quantile(2 / 3))
    top = (adj[m].abs() >= thr) & (adj[m].abs() > 1e-9)
    win = (np.sign(adj[m][top]) == np.sign(d.ats_raw[m][top]))
    return {"n": n, "r": round(r, 4), "t": round(t, 2), "p": float(1 - norm.cdf(t)),
            "top_n": int(top.sum()), "top_hit": round(100 * float(win.mean()), 1) if top.any() else np.nan}


def bh(pvals, q=0.10):
    p = np.asarray(pvals)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.zeros(len(p), bool)
    below = np.where(ranked <= q)[0]
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed


def single_signals(d, tests=range(2015, 2027)):
    tests = list(tests)
    adjs, rows = {}, []
    for c, label in CANDIDATES.items():
        a = oos_adjust(d, [c], tests)
        adjs[c] = a
        s = score(d, a)
        eras = {f"{e0}-{str(e1)[2:]}": score(d, a, d.season.between(e0, e1))["top_hit"] for e0, e1 in ERAS}
        rows.append({"signal": c, "label": label, **s, **eras})
    t = pd.DataFrame(rows)
    t["fdr_pass"] = bh(t.p.values, 0.10)
    t["eras_above_50"] = t[[f"{e0}-{str(e1)[2:]}" for e0, e1 in ERAS]].gt(50).sum(axis=1)
    return t.sort_values("t", ascending=False), adjs


def nested_final(d, adjs, tests=range(2018, 2027), t_min=1.645, min_eras=None):
    """Choose signals per test season from their OOS record in prior seasons only."""
    pred = pd.Series(np.nan, index=d.index)
    chosen = {}
    for S in tests:
        prior = d.season.between(2015, S - 1)
        sel = []
        for c in CANDIDATES:
            sc = score(d, adjs[c], prior)
            if sc["t"] >= t_min:
                sel.append(c)
        chosen[S] = sel
        if sel:
            pred[d.season == S] = oos_adjust(d, sel, [S])[d.season == S]
        else:
            pred[d.season == S] = 0.0
    return pred, chosen


def evaluate(d, adj, tests=range(2018, 2027), thresholds=(0, 0.5, 1.0, 1.5, 2.0), n_boot=2000, seed=7):
    rng = np.random.default_rng(seed)
    m0 = d.season.isin(list(tests)) & adj.notna() & (d.ats_raw != 0)
    rows = []
    for th in thresholds:
        m = m0 & (adj.abs() > th)
        if m.sum() == 0:
            continue
        win = (np.sign(adj[m]) == np.sign(d.ats_raw[m])).astype(float)
        wk = (d.season[m] * 100 + d.week[m]).values
        uw = np.unique(wk)
        idx = {u: np.where(wk == u)[0] for u in uw}
        boots = []
        wv = win.values
        for _ in range(n_boot):
            pick = rng.choice(uw, len(uw))
            ii = np.concatenate([idx[u] for u in pick])
            boots.append(wv[ii].mean())
        lo, hi = np.percentile(boots, [5, 95])
        h = wv.mean()
        rows.append({"edge_gt": th, "bets": int(m.sum()), "bets_per_wk": round(m.sum() / len(np.unique(d.season[m0] * 100 + d.week[m0])), 2),
                     "hit": round(100 * h, 1), "ci90_lo": round(100 * lo, 1), "ci90_hi": round(100 * hi, 1),
                     "roi_110": round(100 * (h * 100 / 110 - (1 - h)), 1), "roi_105": round(100 * (h * 100 / 105 - (1 - h)), 1),
                     "p_vs_52.4": round(binomtest(int(win.sum()), int(m.sum()), 0.524, alternative="greater").pvalue, 4)})
    return pd.DataFrame(rows)
