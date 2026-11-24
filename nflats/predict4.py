"""v4 export: adds player-level matchup context and the expanded signal tests to the week file.
python -m nflats.predict4 2026 12   (after build4 + predict3)"""
import json
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from . import data, validate as V, build4

warnings.filterwarnings("ignore")
NEW = {"unit_match": "Unit matchups (pass/run O vs D)", "deep_match": "Deep passing O vs deep pass D",
       "off_form_diff": "Offense recent form vs season", "def_form_diff": "Defense recent form vs season",
       "avail_rec_diff": "Receivers out (targets x EPA/target)", "avail_rb_diff": "Running backs out (carries x EPA/carry)",
       "avail_pr_diff": "Pass rushers out (sacks/hits)", "avail_cov_diff": "Coverage players out (INT/PBU)",
       "avail_total_diff": "All production-weighted absences"}
WATCH = {"st_ko_diff", "st_punt_diff", "go_rate_diff", "to_diff", "div", "st_fg_diff", "avail_rb_diff"}


def main(season, week):
    wk = json.load(open(f"{data.DATA_DIR}/week3_{season}_{week:02d}.json"))
    x = pd.read_parquet(f"{data.DATA_DIR}/features4.parquet")
    V.CANDIDATES.update(NEW)
    d = V.prep(x)
    t, adjs = V.single_signals(d)
    pickle.dump((d, adjs, t), open(f"{data.DATA_DIR}/v4_signals.pkl", "wb"))
    era_cols = [c for c in t.columns if c[:2] == "20"]
    wk["signals"] = [{"signal": r.signal, "label": r.label, "t": r.t, "p": round(r.p, 4), "top_hit": r.top_hit,
                      "eras": {c: getattr(r, "_asdict", lambda: {})().get(c) for c in era_cols}, "fdr": bool(r.fdr_pass),
                      "verdict": "used" if r.signal == "st_diff" else ("watch" if r.signal in WATCH else "no edge")}
                     for r in t.itertuples()]
    for row, (_, r) in zip(wk["signals"], t.iterrows()):
        row["eras"] = {c: r[c] for c in era_cols}
    wk["signal_labels"] = dict(V.CANDIDATES)

    # this week's players out, valued by production
    p = build4.load(season)
    _, m = build4.availability(season, p)
    cur = m[(m.season == season) & (m.week == week)]
    xw = x[(x.season == season) & (x.week == week)].set_index("game_id")
    for g in wk["games"]:
        r = xw.loc[g["game_id"]]
        num = lambda v: None if pd.isna(v) else round(float(v), 2)
        outs = {}
        for side, team in [("home", g["home"]), ("away", g["away"])]:
            c = cur[cur.team == team].copy()
            c["val"] = c[["rec", "rb", "pr", "cov"]].sum(axis=1)
            c = c[c.val.abs() >= 0.05].sort_values("val", ascending=False).head(6)
            outs[side] = [{"name": n, "pos": ps, "status": s, "val": round(float(v), 2)}
                          for n, ps, s, v in zip(c.full_name, c.position, c.report_status, c.val)]
        g["matchup"] = {"unit_match": num(r.unit_match), "deep_match": num(r.deep_match),
                        "off_form": num(r.off_form_diff), "def_form": num(r.def_form_diff),
                        "qb_blitz": num(r.get("qb_blitz_match")), "lost": {
                            "home": {k: num(r[f"h_lost_{k}"]) or 0 for k in ["rec", "rb", "pr", "cov"]},
                            "away": {k: num(r[f"a_lost_{k}"]) or 0 for k in ["rec", "rb", "pr", "cov"]}},
                        "outs": outs}
    wk["version"] = 4
    path = f"{data.DATA_DIR}/week4_{season}_{week:02d}.json"
    json.dump(wk, open(path, "w"), separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path, wk


if __name__ == "__main__":
    path, w = main(int(sys.argv[1]), int(sys.argv[2]))
    print(path, len(w["signals"]))
    for g in w["games"][:4]:
        print(g["away"], g["home"], g["matchup"])
