"""Walk-forward tests of special-teams v2 candidates with Benjamini-Hochberg FDR (q = 0.10)."""
import json
import os
import pickle
import numpy as np
import pandas as pd
from . import data, validate as V

TESTS = list(range(2016, 2027))


def load():
    d, _, _ = pickle.load(open(f"{data.DATA_DIR}/v3_signals.pkl", "rb"))
    s2 = pd.read_parquet(os.path.join(data.DATA_DIR, "st2.parquet")).drop(columns=["season", "week"])
    d = d.merge(s2, on="game_id", how="left")
    d["ats"] = (d.result - d.spread_line).clip(-21, 21)
    d["ats_raw"] = d.result - d.spread_line
    return d[d.st2_fg.notna()].reset_index(drop=True)


def lean_stats(d, adj, thr=0.5):
    m = d.season.isin(TESTS) & adj.notna() & (adj.abs() >= thr) & (d.ats_raw != 0)
    win = np.sign(adj[m]) == np.sign(d.ats_raw[m])
    return {"n": int(m.sum()), "win": round(100 * float(win.mean()), 1)}


def run():
    d = load()
    base = V.oos_adjust(d, ["st_diff"], TESTS)
    singles = {
        "current: combined special teams": ["st_diff"],
        "split: FG / punt / kickoff weights": ["st2_fg", "st2_punt", "st2_ko"],
        "split + kickoff weight after 2024 rule": ["st2_fg", "st2_punt", "st2_ko", "st2_ko_post24"],
        "kicker skill in wind/cold/altitude": ["kick_val"],
        "game-day punter skill": ["punt_val"],
        "primary returner out": ["ret_out"],
    }
    out, pv = {}, {}
    for name, cols in singles.items():
        adj = V.oos_adjust(d, cols, TESTS)
        sc = V.score(d[d.season.isin(TESTS)], adj[d.season.isin(TESTS)])
        out[name] = {"t": sc["t"], "p": round(sc["p"], 4), "top_third_win": sc["top_hit"], "lean_0.5": lean_stats(d, adj)}
        if name != "current: combined special teams":
            pv["alone: " + name] = sc["p"]
    # increments over the current signal: test each candidate on what the current signal leaves unexplained
    d2 = d.copy()
    d2["ats"] = (d.ats - base.fillna(0)).clip(-21, 21)
    incs = {"kickoff weight after 2024 rule": ["st2_ko_post24"], "kicker skill in wind/cold/altitude": ["kick_val"],
            "game-day punter skill": ["punt_val"], "primary returner out": ["ret_out"],
            "split phases (FG/punt/kickoff)": ["st2_fg", "st2_punt", "st2_ko"]}
    for name, cols in incs.items():
        adj = V.oos_adjust(d2, cols, TESTS)
        sc = V.score(d2[d2.season.isin(TESTS)], adj[d2.season.isin(TESTS)])
        out["on top of current: " + name] = {"t": sc["t"], "p": round(sc["p"], 4)}
        pv["on top of current: " + name] = sc["p"]
    keys = list(pv)
    passed = V.bh([pv[k] for k in keys], q=0.10)
    out["FDR (BH q=0.10) passes"] = [k for k, ok in zip(keys, passed) if ok]
    # combined candidate models, as betting rules (lean >= 0.5)
    for name, cols in {"current only": ["st_diff"], "current + kicker + punter + returner": ["st_diff", "kick_val", "punt_val", "ret_out"],
                       "split + post-2024 kickoff + kicker + punter + returner": ["st2_fg", "st2_punt", "st2_ko", "st2_ko_post24", "kick_val", "punt_val", "ret_out"]}.items():
        adj = V.oos_adjust(d, cols, TESTS)
        sc = V.score(d[d.season.isin(TESTS)], adj[d.season.isin(TESTS)])
        out["model: " + name] = {"t": sc["t"], "lean_0.5": lean_stats(d, adj), "lean_0.5_2024+": lean_stats(d[d.season >= 2024], adj[d.season >= 2024])}
    return out


if __name__ == "__main__":
    for k, v in run().items():
        print(k, json.dumps(v))
