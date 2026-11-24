"""Backtest using the downloaded line history (python -m nflats.linehistory FILE.json).

Tests (all walk-forward; special-teams weight learned only from earlier seasons):
  1. How lines move: open -> Friday -> close, and how often they cross 3 and 7
  2. Closing-line value of the model's special-teams lean (does the market move toward it?)
  3. Betting the model at the opener vs at the close (hit rate, ROI at -110 and at best available price)
  4. Line moves as signals: early move (open->Fri), full move (open->close), total move
  5. US books off Pinnacle at kickoff: bet the stale number?
"""
import json
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from . import data, validate as V, model as M

warnings.filterwarnings("ignore")
PAY = lambda o: o / 100 if o > 0 else 100 / abs(o)


def load(path):
    j = json.load(open(path))
    rows = []
    for sid, sn in j["snaps"].items():
        for gid, books in sn["g"].items():
            for bk, (L, oh, oa, tot) in books.items():
                rows.append((gid, sn["k"], sid, bk, L, oh, oa, tot))
    df = pd.DataFrame(rows, columns=["game_id", "kind", "snap", "book", "L", "oh", "oa", "tot"])
    return df


def per_game(df):
    g = df.groupby(["game_id", "kind"]).agg(L=("L", "median"), tot=("tot", "median")).unstack("kind")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    pin = df[df.book == "pinnacle"].groupby(["game_id", "kind"]).L.median().unstack("kind").add_prefix("pin_")
    return g.join(pin, how="left").reset_index()


def best_price(df, kind):
    """Best line for each side at a stage: highest L for away bettors? (L = expected home margin)
    Home bettor wants the LOWEST L (home gets more points); away bettor wants the HIGHEST L."""
    d = df[(df.kind == kind) & df.L.notna()]
    h = d.sort_values(["game_id", "L", "oh"], ascending=[True, True, False]).groupby("game_id").first()[["L", "oh", "book"]]
    a = d.sort_values(["game_id", "L", "oa"], ascending=[True, False, False]).groupby("game_id").first()[["L", "oa", "book"]]
    return h.add_prefix("h_"), a.add_prefix("a_")


def stats(won, push=None, odds=None):
    won = pd.Series(won).astype(bool)
    n = len(won)
    if push is not None:
        keep = ~pd.Series(push).astype(bool).values
        won = won[keep]
        odds = None if odds is None else pd.Series(odds)[keep]
    n = len(won)
    if n == 0:
        return {"n": 0}
    h = won.mean()
    out = {"n": n, "hit": round(100 * h, 1), "roi_110": round(100 * (h * 100 / 110 - (1 - h)), 1),
           "p_vs_52.4": round(binomtest(int(won.sum()), n, 0.524, alternative="greater").pvalue, 3)}
    if odds is not None:
        pnl = np.where(won, [PAY(o) for o in odds], -1.0)
        out["roi_best_price"] = round(100 * float(np.mean(pnl)), 1)
    return out


def main(path):
    raw = load(path)
    G = per_game(raw)
    d, adjs, _ = pickle.load(open(f"{data.DATA_DIR}/v3_signals.pkl", "rb"))
    tests = sorted(d.season.unique())
    tests = [s for s in tests if s >= 2016]
    st = V.oos_adjust(d, ["st_diff"], tests)
    base = d.assign(st_adj=st, stats_line=d.spread_line + d.stats_gap)[["game_id", "season", "week", "result", "spread_line", "total_line", "st_adj", "wind_mph", "stats_line"]]
    X = base.merge(G, on="game_id", how="inner")
    hb, ab = best_price(raw, "open")
    X = X.merge(hb, left_on="game_id", right_index=True, how="left").merge(ab, left_on="game_id", right_index=True, how="left")
    X = X[X.L_open.notna() & X.L_close.notna() & X.result.notna()]
    R = {"games": int(len(X)), "seasons": sorted(int(s) for s in X.season.unique())}

    # 1. movement
    mv = X.L_close - X.L_open
    cross = lambda a, b, k: ((np.minimum(a.abs(), b.abs()) < k) & (np.maximum(a.abs(), b.abs()) > k))
    R["movement"] = {"mean_abs_move": round(float(mv.abs().mean()), 2), "moved_0.5+": round(100 * float((mv.abs() >= 0.5).mean()), 1),
                     "moved_1.5+": round(100 * float((mv.abs() >= 1.5).mean()), 1),
                     "crossed_3": round(100 * float(cross(X.L_open, X.L_close, 3).mean()), 1),
                     "crossed_7": round(100 * float(cross(X.L_open, X.L_close, 7).mean()), 1),
                     "close_vs_result_mae": round(float((X.result - X.L_close).abs().mean()), 2),
                     "open_vs_result_mae": round(float((X.result - X.L_open).abs().mean()), 2)}

    # 2. closing line value of the special-teams lean
    lean = np.sign(X.st_adj)
    clv = lean * (X.L_close - X.L_open)  # + = market moved toward our side
    strong = X.st_adj.abs() >= 0.5
    R["clv"] = {"all_leans_avg_pts": round(float(clv[lean != 0].mean()), 3),
                "strong_leans_avg_pts": round(float(clv[strong].mean()), 3),
                "strong_moved_toward_%": round(100 * float((clv[strong] > 0).mean()), 1),
                "strong_moved_against_%": round(100 * float((clv[strong] < 0).mean()), 1), "n_strong": int(strong.sum())}

    # 3. bet the lean at open vs at close
    for stage in ["open", "fri", "close"]:
        col = f"L_{stage}"
        m = strong & X[col].notna()
        ats = (X.result - X[col])[m] * lean[m]
        R[f"ST_lean_at_{stage}"] = stats(ats > 0, ats == 0)
    # best available price at open
    m = strong & X.h_L.notna() & X.a_L.notna()
    side_home = X.st_adj > 0
    line = np.where(side_home, X.h_L, X.a_L)
    odds = np.where(side_home, X.h_oh, X.a_oa)
    ats = np.where(side_home, X.result - line, line - X.result)
    R["ST_lean_best_book_at_open"] = stats(pd.Series(ats[m]) > 0, pd.Series(ats[m]) == 0, pd.Series(odds[m]).fillna(-110))

    # 3b. full stats model vs the opener (it loses to the close; does it beat the open?)
    for th in [1.5, 3.0]:
        for stage in ["open", "close"]:
            gap = X.stats_line - X[f"L_{stage}"]
            m = (gap.abs() >= th) & X.stats_line.notna()
            a = (X.result - X[f"L_{stage}"])[m] * np.sign(gap[m])
            R[f"stats_model_gap{th}_at_{stage}"] = stats(a > 0, a == 0)
        gap = X.stats_line - X.L_open
        m = (gap.abs() >= th) & X.stats_line.notna()
        c = np.sign(gap[m]) * (X.L_close - X.L_open)[m]
        R[f"stats_model_gap{th}_clv"] = {"avg_pts": round(float(c.mean()), 3), "toward_%": round(100 * float((c > 0).mean()), 1),
                                         "against_%": round(100 * float((c < 0).mean()), 1), "n": int(m.sum())}
    # per-season for the special-teams lean at open
    R["ST_lean_at_open_by_season"] = {int(S): stats(((X.result - X.L_open) * lean)[strong & (X.season == S)] > 0,
                                                     ((X.result - X.L_open) * lean)[strong & (X.season == S)] == 0)["hit"]
                                      for S in sorted(X.season.unique())}

    # EV tiers at open (projection = opening consensus + special-teams adjustment), key-number odds
    rows = []
    for S in sorted(X.season.unique()):
        T = M.pmf_tables_by_total(d, S)
        for r in X[X.season == S].itertuples():
            if pd.isna(r.h_L) or pd.isna(r.a_L):
                continue
            et = M.effective_total(r.total_line, r.wind_mph)
            tab = T["lo" if et <= M.TOTAL_SPLIT else "hi"]
            mu = r.L_open + r.st_adj
            hh, hp, ha = M.cover_probs(mu, r.h_L, tab)
            ah, ap, aa = M.cover_probs(mu, r.a_L, tab)
            evh = hh * PAY(r.h_oh if r.h_oh == r.h_oh else -110) - ha
            eva = aa * PAY(r.a_oa if r.a_oa == r.a_oa else -110) - ah
            if evh >= eva:
                res = r.result - r.h_L; o = r.h_oh
            else:
                res = r.a_L - r.result; o = r.a_oa
            rows.append((max(evh, eva), res, o if o == o else -110))
    E = pd.DataFrame(rows, columns=["ev", "res", "odds"])
    for lab, lo, hi in [("A_open_EV5+", 0.05, 9), ("B_open_EV3-5", 0.03, 0.05), ("open_EV0-3", 0, 0.03)]:
        e = E[(E.ev >= lo) & (E.ev < hi)]
        R[lab] = stats(e.res > 0, e.res == 0, e.odds)

    # 4. line moves as signals (vs closing line, so no lookahead in the bet itself)
    ats_close = X.result - X.L_close
    for name, sig, mask in [("early_move_open_to_fri", X.L_fri - X.L_open, X.L_fri.notna()),
                            ("late_move_fri_to_close", X.L_close - X.L_fri, X.L_fri.notna()),
                            ("full_move_open_to_close", X.L_close - X.L_open, X.L_close.notna())]:
        m = mask & (sig.abs() >= 0.5)
        s = np.sign(sig[m])
        a = ats_close[m] * s
        R[f"follow_{name}"] = stats(a > 0, a == 0)
    # does the early move continue? (open->fri predicts fri->close)
    m = X.L_fri.notna() & ((X.L_fri - X.L_open).abs() >= 0.5)
    cont = np.sign(X.L_fri - X.L_open)[m] * (X.L_close - X.L_fri)[m]
    R["early_move_continues_%"] = round(100 * float((cont > 0).mean()), 1) if m.any() else None
    fav = np.sign(X.L_close).replace(0, 1)
    tm = (X.tot_close - X.tot_open)
    m = tm.abs() >= 1
    a = ats_close[m] * fav[m] * np.sign(tm[m])
    R["total_move_with_favorite"] = stats(a > 0, a == 0)

    # 5. US books off Pinnacle at kickoff
    cl = raw[raw.kind == "close"].merge(X[["game_id", "result"]], on="game_id")
    pin = cl[cl.book == "pinnacle"][["game_id", "L"]].rename(columns={"L": "pinL"})
    cl = cl[cl.book != "pinnacle"].merge(pin, on="game_id")
    cl = cl[cl.L.notna() & cl.pinL.notna()]
    gap = cl.L - cl.pinL  # + = book more home-favoured than Pinnacle -> away bettor gets more points
    cl["gap"] = cl.L - cl.pinL
    cl["side"] = np.where(cl.gap > 0, -1, 1)   # take the side the book is giving extra points to
    cl["odds"] = np.where(cl.side > 0, cl.oh, cl.oa)
    cl["odds"] = cl.odds.fillna(-110)
    cl["absgap"] = cl.gap.abs()
    out = []
    for th in [0.5, 1.0, 1.5]:
        c = cl[cl.absgap >= th].sort_values(["game_id", "absgap", "odds"], ascending=[True, False, False]).groupby("game_id").first()
        res = (c.result - c.L) * c.side
        out.append((th, stats(res > 0, res == 0, c.odds)))
    sub = cl[cl.absgap >= 0.5]
    R["off_pinnacle_by_book"] = {bk: stats(((g.result - g.L) * g.side) > 0, ((g.result - g.L) * g.side) == 0, g.odds)
                                 for bk, g in sub.groupby("book")}
    R["rec_books_off_pinnacle"] = {f"gap>={th}": s for th, s in out}
    # same at the opener
    op = raw[raw.kind == "open"].merge(X[["game_id", "result"]], on="game_id")
    pin = op[op.book == "pinnacle"][["game_id", "L"]].rename(columns={"L": "pinL"})
    op = op[op.book != "pinnacle"].merge(pin, on="game_id")
    op = op[op.L.notna() & op.pinL.notna()]
    op["gap"] = op.L - op.pinL; op["side"] = np.where(op.gap > 0, -1, 1)
    op["odds"] = np.where(op.side > 0, op.oh, op.oa); op["odds"] = op.odds.fillna(-110); op["absgap"] = op.gap.abs()
    o2 = {}
    for th in [0.5, 1.0, 1.5]:
        c = op[op.absgap >= th].sort_values(["game_id", "absgap", "odds"], ascending=[True, False, False]).groupby("game_id").first()
        res = (c.result - c.L) * c.side
        o2[f"gap>={th}"] = stats(res > 0, res == 0, c.odds)
    R["rec_books_off_pinnacle_at_open"] = o2
    # Pinnacle's own move vs US consensus move (does Pinnacle lead?)
    pm = X.pin_close - X.pin_open
    m = pm.abs() >= 0.5
    a = (X.result - X.L_open)[m] * np.sign(pm[m])
    R["bet_US_open_in_direction_of_pinnacle_move"] = stats(a > 0, a == 0)
    return R


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))
