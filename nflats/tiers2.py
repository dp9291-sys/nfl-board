"""Special-teams-first tiers, priced with the smooth key-number model (walk-forward).

Rule (fixed before testing):
  candidate  : special-teams adjustment (c_st x st_diff, weight learned from earlier seasons) >= 0.5 pts
  side       : the team special teams favours
  price check: smooth-model expected return at the available line and price >= 0
               (projection = market line + special-teams adjustment)
  Tier A     : lean >= 1.0 pt and passes the price check
  Tier B     : lean 0.5-1.0 pt and passes the price check
  stakes     : Kelly fraction x assumed edge (A 5%, B 3%) / payout, capped at 2% of bankroll
"""
import json
import pickle
import sys
import numpy as np
import pandas as pd
from . import data, validate as V, cover2 as C

PAY = lambda o: o / 100 if o > 0 else 100 / abs(o)
EDGE = {"A": 0.05, "B": 0.03}


def classify(st_pts, ev, floor=0.0):
    a = abs(st_pts)
    if a < 0.5 or ev < floor:
        return "pass"
    return "A" if a >= 1.0 else "B"


def stake_frac(tier, odds, kelly=0.25, cap=0.02):
    if tier not in EDGE:
        return 0.0
    return min(cap, kelly * EDGE[tier] / PAY(odds))


def side_ev(mu, line, odds_side, side, par):
    h, p, a = C.cover_probs(mu, line, par)
    return h * PAY(odds_side) - a if side == "home" else a * PAY(odds_side) - h


def rows_posted(d, st_col="st_adj", seasons=range(2018, 2027), games=None, floor=0.0):
    games = games if games is not None else data.load_games()
    out = []
    for S in seasons:
        P = C.fit(games, S)
        X = d[(d.season == S) & d.result.notna() & d.spread_line.notna() & d[st_col].notna()]
        for r in X.itertuples():
            st = getattr(r, st_col)
            side = "home" if st > 0 else "away"
            par = C.params(P, C.group(r.total_line, r.wind_mph), r.spread_line)
            o = r.home_spread_odds if side == "home" else r.away_spread_odds
            o = o if o == o else -110
            ev = side_ev(r.spread_line + st, r.spread_line, o, side, par)
            res = (r.result - r.spread_line) * (1 if side == "home" else -1)
            out.append((S, st, ev, classify(st, ev, floor), res, o))
    return pd.DataFrame(out, columns=["S", "st", "ev", "tier", "res", "odds"])


def summarize(E, label):
    def s(e):
        dec = e[e.res != 0]
        pnl = np.where(e.res > 0, [PAY(o) for o in e.odds], np.where(e.res < 0, -1.0, 0.0))
        return {"n": int(len(e)), "win": round(100 * float((dec.res > 0).mean()), 1) if len(dec) else None,
                "roi": round(100 * float(pnl.mean()), 1) if len(e) else None}
    out = {"label": label, "A": s(E[E.tier == "A"]), "B": s(E[E.tier == "B"]), "A+B": s(E[E.tier.isin(["A", "B"])]),
           "ST lean 0.5+ (no price check)": s(E[E.st.abs() >= 0.5]),
           "price check failed": s(E[(E.st.abs() >= 0.5) & (E.tier == "pass")])}
    eras = {}
    for a, b in [(2018, 2020), (2021, 2023), (2024, 2026)]:
        e = E[(E.S >= a) & (E.S <= b) & E.tier.isin(["A", "B"])]
        eras[f"{a}-{str(b)[2:]}"] = s(e)
    out["A+B by era"] = eras
    # bankroll sim: 1/4 Kelly at assumed edges, capped 2%, flat-bank sizing per bet (no compounding)
    e = E[E.tier.isin(["A", "B"])]
    fr = np.array([stake_frac(t, o) for t, o in zip(e.tier, e.odds)])
    pnl = np.where(e.res > 0, [PAY(o) for o in e.odds], np.where(e.res < 0, -1.0, 0.0))
    out["units_at_quarter_kelly_pct_bankroll"] = round(float(100 * np.sum(fr * pnl)), 1)
    return out


def main():
    d, _, _ = pickle.load(open(f"{data.DATA_DIR}/v3_signals.pkl", "rb"))
    tests = [s for s in sorted(d.season.unique()) if s >= 2016]
    d = d.assign(st_adj=V.oos_adjust(d, ["st_diff"], tests))
    games = data.load_games()
    for floor in [0.0, -0.02, -9]:
        E = rows_posted(d, games=games, floor=floor)
        lab = {0.0: "price check EV >= 0 (primary)", -0.02: "sensitivity: EV >= -2%", -9: "sensitivity: no price check"}[floor]
        print(json.dumps(summarize(E, lab)))


if __name__ == "__main__":
    main()
