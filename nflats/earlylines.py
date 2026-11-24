"""Earlier-lines tests (python -m nflats.earlylines FILE.json [MORE.json ...]); files are merged.

Rules fixed before the data arrived:
 primary  : special-teams lean (>= 0.5 pts, weight learned from earlier seasons) bet at the best Arizona book
            on the Sunday-night line ('sun'), vs the same bets near kickoff ('close')
 secondary: same at the look-ahead line ('ahead'); full stats line 3+ pts off each early line (ATS and how often
            the line later moves toward it); accuracy of each stage's consensus line
"""
import json
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from . import data, validate as V, linehistory as LH

AZ = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet"]
STAGES = ["ahead", "sun", "open", "fri", "close"]


def merge(paths):
    out = {"snaps": {}}
    for p in paths:
        j = json.load(open(p))
        out["snaps"].update(j["snaps"])
        out.setdefault("games", j.get("games"))
    tmp = "/tmp/_merged_history.json"
    json.dump(out, open(tmp, "w"))
    return tmp


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
        pnl = np.where(res > 0, np.where(od > 0, od / 100, 100 / np.abs(od)), np.where(res < 0, -1.0, 0.0))
        o["roi"] = round(100 * float(pnl.mean()), 1)
    return o


def main(paths):
    raw = LH.load(merge(paths))
    G = LH.per_game(raw)
    d, _, _ = pickle.load(open(f"{data.DATA_DIR}/v3_signals.pkl", "rb"))
    tests = [s for s in sorted(d.season.unique()) if s >= 2016]
    d = d.assign(st_adj=V.oos_adjust(d, ["st_diff"], tests), stats_line=d.spread_line + d.stats_gap)
    X = d.merge(G, on="game_id")
    X = X[X.result.notna()]
    R = {"games": int(len(X)), "coverage_by_stage": {s: int(X[f"L_{s}"].notna().sum()) for s in STAGES if f"L_{s}" in X}}
    R["line_accuracy_mae"] = {s: round(float((X.result - X[f"L_{s}"]).abs()[X[f"L_{s}"].notna()].mean()), 3)
                              for s in STAGES if f"L_{s}" in X}
    st = X[X.st_adj.abs() >= 0.5]
    for s in STAGES:
        if f"L_{s}" not in X:
            continue
        h, a = LH.best_price(raw[raw.book.isin(AZ)], s)
        Y = st.merge(h, left_on="game_id", right_index=True).merge(a, left_on="game_id", right_index=True)
        hm = Y.st_adj > 0
        L = np.where(hm, Y.h_L, Y.a_L)
        o = np.where(hm, Y.h_oh, Y.a_oa)
        res = np.where(hm, Y.result - L, L - Y.result)
        R[f"special teams at best AZ book, {s}" + (" (primary)" if s == "sun" else "")] = rec(res, o)
        mv = (np.sign(st.st_adj) * (st.L_close - st[f"L_{s}"]))[st[f"L_{s}"].notna()]
        R[f"  line moved toward special-teams side after {s} (toward/against %)"] = [round(100 * float((mv > 0).mean()), 1),
                                                                                     round(100 * float((mv < 0).mean()), 1)]
    for s in ["ahead", "sun", "open"]:
        if f"L_{s}" not in X:
            continue
        gap = X.stats_line - X[f"L_{s}"]
        m = (gap.abs() >= 3) & X[f"L_{s}"].notna() & X.stats_line.notna()
        res = (X.result - X[f"L_{s}"])[m] * np.sign(gap[m])
        mv = (np.sign(gap[m]) * (X.L_close - X[f"L_{s}"])[m])
        R[f"full stats line 3+ off the {s} line"] = {**rec(res), "moved_toward_%": round(100 * float((mv > 0).mean()), 1),
                                                     "moved_against_%": round(100 * float((mv < 0).mean()), 1)}
    return R


if __name__ == "__main__":
    for k, v in main(sys.argv[1:]).items():
        print(k, json.dumps(v))
