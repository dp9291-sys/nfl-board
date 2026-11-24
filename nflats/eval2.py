"""Phase 2 evaluation (python -m nflats.eval2 LINE_HISTORY.json). Rules fixed before running.

A accuracy     : residual MAE / RMSE of close + ST adj vs close alone; Brier + log loss of cover odds vs a coin flip
B stakes       : on ST bets, out-of-sample Brier of (i) raw model cover odds, (ii) flat rate learned from earlier
                 seasons, (iii) walk-forward logistic recalibration of (i); bankroll growth of flat vs EV stakes
C Pinnacle     : ST bets graded against Pinnacle's closing line; best Arizona line vs Pinnacle close
D dispersion   : ST adj x book-line spread at close (interaction, walk-forward)
E challengers  : ST-B opponent-adjusted (weekly ridge), ST-C per-play; vs incumbent on same games
F wind         : ST adj x outdoor 10+ mph wind (interaction, walk-forward)
FDR (BH q=0.10) over the yes/no tests: B(iii) beats flat, D, E-B, E-C, F
"""
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp
from . import data, validate as V, cover2 as C, linehistory as LH, st2
from .ratings import TEAMS, TIDX, _weights

AZ = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet"]
TESTS = list(range(2016, 2027))


def base_frame():
    d, _, _ = pickle.load(open(os.path.join(data.DATA_DIR, "v3_signals.pkl"), "rb"))
    d = d[d.result.notna() & d.spread_line.notna()].reset_index(drop=True)
    d["ats"] = (d.result - d.spread_line).clip(-21, 21)
    d["ats_raw"] = d.result - d.spread_line
    d["st_adj"] = V.oos_adjust(d, ["st_diff"], TESTS)
    return d


def cover_cols(d):
    games = data.load_games()
    P = {S: C.fit(games, S) for S in TESTS}
    ph, pp = [], []
    for r in d.itertuples():
        if r.season not in P or r.st_adj != r.st_adj:
            ph.append(np.nan); pp.append(np.nan); continue
        par = C.params(P[r.season], C.group(r.total_line, r.wind_mph), r.spread_line)
        h, p, a = C.cover_probs(r.spread_line + r.st_adj, r.spread_line, par)
        ph.append(h / (h + a)); pp.append(p)
    d["p_home"] = ph
    return d


def brier_ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return round(float(np.mean((p - y) ** 2)), 5), round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5)


def test_A(d):
    m = d.season.isin(TESTS) & d.st_adj.notna()
    x = d[m]
    e0, e1 = x.result - x.spread_line, x.result - (x.spread_line + x.st_adj)
    dec = x[x.ats_raw != 0]
    y = (dec.ats_raw > 0).astype(float)
    diff = np.abs(e0) - np.abs(e1)
    return {"games": int(len(x)), "MAE close": round(float(e0.abs().mean()), 4), "MAE close+ST": round(float(e1.abs().mean()), 4),
            "MAE improvement pts": round(float(diff.mean()), 4), "improvement t": round(float(ttest_1samp(diff, 0).statistic), 2),
            "RMSE close / close+ST": [round(float(np.sqrt((e0 ** 2).mean())), 3), round(float(np.sqrt((e1 ** 2).mean())), 3)],
            "bias close+ST (mean residual)": round(float(e1.mean()), 3),
            "Brier/logloss coin flip": brier_ll(np.full(len(y), 0.5), y), "Brier/logloss model": brier_ll(dec.p_home, y)}


def test_B(d):
    x = d[d.season.isin(TESTS) & (d.st_adj.abs() >= 0.5) & (d.ats_raw != 0)].copy()
    side = np.sign(x.st_adj)
    x["win"] = (np.sign(x.ats_raw) == side).astype(float)
    x["p_raw"] = np.where(side > 0, x.p_home, 1 - x.p_home)
    # flat and recalibrated, walk-forward by season (train on earlier test seasons; first season uses 0.524)
    flat, cal = [], []
    for r in x.itertuples():
        tr = x[x.season < r.season]
        if len(tr) < 100:
            flat.append(0.524); cal.append(r.p_raw); continue
        flat.append(float(tr.win.mean()))
        z = np.log(tr.p_raw / (1 - tr.p_raw))
        A = np.c_[np.ones(len(z)), z]
        b = np.zeros(2)
        for _ in range(25):
            p = 1 / (1 + np.exp(-(A @ b)))
            W = p * (1 - p)
            b += np.linalg.solve(A.T @ (A * W[:, None]) + 1e-3 * np.eye(2), A.T @ (tr.win - p))
        zz = np.log(r.p_raw / (1 - r.p_raw))
        cal.append(float(1 / (1 + np.exp(-(b[0] + b[1] * zz)))))
    x["p_flat"], x["p_cal"] = flat, cal
    out = {"bets": int(len(x)), "win rate": round(100 * float(x.win.mean()), 1),
           "Brier/logloss raw model": brier_ll(x.p_raw, x.win), "Brier/logloss flat": brier_ll(x.p_flat, x.win),
           "Brier/logloss recalibrated": brier_ll(x.p_cal, x.win)}
    dl = (x.p_flat - x.win) ** 2 - (x.p_cal - x.win) ** 2       # >0 means recalibrated is better
    t = float(ttest_1samp(dl, 0).statistic)
    out["recalibrated vs flat: Brier gain t"] = round(t, 2)
    out["_p"] = float(1 - norm.cdf(t))
    # does the raw model rank bets? win rate by model-probability tercile
    q = pd.qcut(x.p_raw, 3, labels=["low", "mid", "high"])
    out["win rate by model-probability third"] = {k: round(100 * float(g.win.mean()), 1) for k, g in x.groupby(q, observed=True)}
    # bankroll growth (flat bankroll units, 1/4 Kelly, 2% cap) at -110
    b_ = 100 / 110
    def growth(p):
        f = np.clip(0.25 * (p * (b_ + 1) - 1) / b_, 0, 0.02)
        return float(np.sum(np.log1p(np.where(x.win > 0, f * b_, -f))))
    out["log-growth: flat 2% edge stakes"] = round(float(np.sum(np.log1p(np.where(x.win > 0, 0.0055 * b_, -0.0055)))), 4)
    out["log-growth: recalibrated EV stakes"] = round(growth(x.p_cal.values), 4)
    out["log-growth: raw model EV stakes"] = round(growth(x.p_raw.values), 4)
    return out


def test_C(d, raw):
    G = LH.per_game(raw)
    x = d.merge(G, on="game_id")
    x = x[x.pin_close.notna() & (x.st_adj.abs() >= 0.5)]
    side = np.sign(x.st_adj)
    res_pin = (x.result - x.pin_close) * side
    hb, ab = LH.best_price(raw[raw.book.isin(AZ)], "close")
    y = x.merge(hb, left_on="game_id", right_index=True).merge(ab, left_on="game_id", right_index=True)
    s2 = np.sign(y.st_adj)
    L = np.where(s2 > 0, y.h_L, y.a_L)
    gain = np.where(s2 > 0, y.pin_close - L, L - y.pin_close)   # points better than Pinnacle close for our side
    dec = res_pin[res_pin != 0]
    return {"ST bets graded at Pinnacle close": {"n": int(len(dec)), "win": round(100 * float((dec > 0).mean()), 1)},
            "ST bets graded at consensus close": {"win": round(100 * float(((x.result - x.L_close) * side)[(x.result - x.L_close) != 0].gt(0).mean()), 1)},
            "best Arizona line vs Pinnacle close (avg pts better for our side)": round(float(np.mean(gain)), 3),
            "share of bets where best AZ line beats Pinnacle close": round(100 * float(np.mean(gain > 0)), 1)}


def interaction_test(d, col):
    x = d[d.season.isin(TESTS) & d.st_adj.notna() & d[col].notna()].copy()
    x["ix"] = x.st_adj * x[col]
    base = x.st_adj
    y = x.copy()
    y["ats"] = (x.ats - base).clip(-21, 21)
    adj = V.oos_adjust(y.reset_index(drop=True), ["ix"], [s for s in TESTS if s > 2016])
    yy = y.reset_index(drop=True)
    m = yy.season.isin([s for s in TESTS if s > 2016])
    sc = V.score(yy[m], adj[m])
    return sc


def test_D(d, raw):
    cl = raw[(raw.kind == "close") & raw.L.notna()]
    disp = cl.groupby("game_id").L.agg(lambda s: float(s.max() - s.min()))
    x = d.merge(disp.rename("disp"), left_on="game_id", right_index=True, how="left")
    x["disp_hi"] = (x.disp >= 1.0).astype(float).where(x.disp.notna())
    st = x[(x.st_adj.abs() >= 0.5) & x.disp.notna() & (x.ats_raw != 0)]
    w = np.sign(st.st_adj) == np.sign(st.ats_raw)
    out = {"ST bets, books within 0.5 pt": {"n": int((st.disp <= 0.5).sum()), "win": round(100 * float(w[st.disp <= 0.5].mean()), 1)},
           "ST bets, books 1+ pt apart": {"n": int((st.disp >= 1).sum()), "win": round(100 * float(w[st.disp >= 1].mean()), 1)}}
    xx = x[x.season >= 2021].copy()
    xx["disp_z"] = (xx.disp - xx.disp.mean()) / xx.disp.std()
    sub = xx[xx.disp_z.notna()].reset_index(drop=True)
    sub["ix"] = sub.st_adj * sub.disp_z
    sub["ats"] = (sub.ats - sub.st_adj).clip(-21, 21)
    adj = V.oos_adjust(sub, ["ix"], [2022, 2023, 2024, 2025, 2026], first=2021)
    m = sub.season >= 2022
    sc = V.score(sub[m], adj[m])
    out["interaction t (2022-26)"] = sc["t"]
    out["_p"] = sc["p"]
    return out


def st_challengers(last=2026):
    cols = ["game_id", "season", "week", "posteam", "defteam", "home_team", "away_team", "special_teams_play", "epa"]
    pbp = pd.concat([pd.read_parquet(os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"), columns=cols) for y in range(2011, last + 1)], ignore_index=True)
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        pbp[c] = pbp[c].replace(data.TEAM_MAP)
    st = pbp[(pbp.special_teams_play == 1) & pbp.epa.notna() & pbp.posteam.notna()]
    a = st.groupby(["game_id", "posteam"]).epa.sum()
    b = st.groupby(["game_id", "defteam"]).epa.sum().rename_axis(["game_id", "posteam"])
    n = st.groupby("game_id").size().rename("n_st")
    keys = pbp[pbp.posteam.notna()].groupby(["game_id", "posteam"]).agg(season=("season", "first"), week=("week", "first")).reset_index()
    tg = keys.merge(a.sub(b, fill_value=0).rename("st").reset_index(), on=["game_id", "posteam"], how="left").fillna({"st": 0})
    tg = tg.merge(n, left_on="game_id", right_index=True, how="left")
    lg_n = float(tg.drop_duplicates("game_id").n_st.mean())
    tg["st_pp"] = tg.st / tg.n_st * lg_n          # per-play value scaled to a typical game's number of ST plays
    hm = pbp.drop_duplicates("game_id").set_index("game_id")[["home_team", "away_team"]]
    g = data.load_games()
    g = g[(g.season >= 2013) & (g.season <= last)]
    rows = []
    for (S, W), wk in g.sort_values(["season", "week"]).groupby(["season", "week"]):
        A = st2.decayed(tg, "st", S, W, K=4.0)
        Cc = st2.decayed(tg, "st_pp", S, W, K=4.0)
        # opponent-adjusted: weighted ridge on home-row net values: y = r_home - r_away
        prev_max = tg.loc[tg.season == S - 1, "week"].max()
        m = ((tg.season == S) & (tg.week < W)) | (tg.season == S - 1)
        win = tg[m]
        win = win[win.game_id.isin(hm.index)]
        win = win[win.posteam == win.game_id.map(hm.home_team)]
        if len(win) > 20:
            w = _weights(win, S, W, prev_max if prev_max == prev_max else 18)
            hi = win.game_id.map(hm.home_team).map(TIDX).values
            ai = win.game_id.map(hm.away_team).map(TIDX).values
            ok = ~(pd.isna(hi) | pd.isna(ai))
            X = np.zeros((ok.sum(), len(TEAMS)))
            X[np.arange(ok.sum()), hi[ok].astype(int)] = 1
            X[np.arange(ok.sum()), ai[ok].astype(int)] = -1
            yv, wv = win.st.values[ok], w[ok]
            Bm = np.linalg.solve(X.T @ (X * wv[:, None]) + 4.0 * np.eye(len(TEAMS)), X.T @ (wv * yv))
        else:
            Bm = np.zeros(len(TEAMS))
        for r in wk.itertuples():
            if r.home_team in TIDX and r.away_team in TIDX:
                h, aa = TIDX[r.home_team], TIDX[r.away_team]
                rows.append((r.game_id, A[h] - A[aa], Bm[h] - Bm[aa], Cc[h] - Cc[aa]))
    return pd.DataFrame(rows, columns=["game_id", "stA", "stB", "stC"])


def test_E(d):
    ch = st_challengers()
    x = d.merge(ch, on="game_id").reset_index(drop=True)
    out, pv = {}, {}
    m = x.season.isin(TESTS)
    for name, col in [("incumbent (ST-A, per game)", "stA"), ("ST-B opponent-adjusted", "stB"), ("ST-C per play", "stC")]:
        adj = V.oos_adjust(x, [col], TESTS)
        sc = V.score(x[m], adj[m])
        e1 = (x.result - (x.spread_line + adj))[m]
        out[name] = {"t": sc["t"], "residual MAE": round(float(e1.abs().mean()), 4),
                     "lean 0.5 win": round(100 * float((np.sign(adj[m & (adj.abs() >= .5) & (x.ats_raw != 0)]) ==
                                                         np.sign(x.ats_raw[m & (adj.abs() >= .5) & (x.ats_raw != 0)])).mean()), 1)}
        if col != "stA":
            y = x.copy()
            base = V.oos_adjust(x, ["stA"], TESTS).fillna(0)
            y["ats"] = (x.ats - base).clip(-21, 21)
            a2 = V.oos_adjust(y, [col], TESTS)
            s2 = V.score(y[m], a2[m])
            out[name]["on top of incumbent t"] = s2["t"]
            pv[name] = s2["p"]
    out["corr with incumbent"] = {"ST-B": round(float(x.stA.corr(x.stB)), 3), "ST-C": round(float(x.stA.corr(x.stC)), 3)}
    return out, pv


def test_F(d):
    g = data.load_games()[["game_id", "roof", "wind"]].rename(columns={"wind": "wind_g", "roof": "roof_g"})
    x = d.merge(g, on="game_id", how="left")
    x["roof"] = x.roof_g
    x["windy"] = ((x.roof.isin(["outdoors", "open"])) & (x.wind_g.fillna(x.wind_mph) >= 10)).astype(float)
    st = x[(x.st_adj.abs() >= 0.5) & (x.ats_raw != 0) & x.season.isin(TESTS)]
    w = np.sign(st.st_adj) == np.sign(st.ats_raw)
    out = {"ST bets in 10+ mph wind": {"n": int((st.windy == 1).sum()), "win": round(100 * float(w[st.windy == 1].mean()), 1)},
           "ST bets otherwise": {"n": int((st.windy == 0).sum()), "win": round(100 * float(w[st.windy == 0].mean()), 1)}}
    sc = interaction_test(x.reset_index(drop=True), "windy")
    out["interaction t"] = sc["t"]
    out["_p"] = sc["p"]
    return out


def main(path):
    d = cover_cols(base_frame())
    raw = LH.load(path)
    R = {"A accuracy vs the closing line": test_A(d), "B EV stakes vs flat": test_B(d), "C Pinnacle benchmark": test_C(d, raw),
         "D market dispersion": test_D(d, raw)}
    e, pv = test_E(d)
    R["E special-teams challengers"] = e
    R["F special teams x wind"] = test_F(d)
    fam = {"B recalibrated beats flat": R["B EV stakes vs flat"].pop("_p"), "D dispersion interaction": R["D market dispersion"].pop("_p"),
           "E ST-B on top": pv["ST-B opponent-adjusted"], "E ST-C on top": pv["ST-C per play"], "F wind interaction": R["F special teams x wind"].pop("_p")}
    ok = V.bh(list(fam.values()), q=0.10)
    R["FDR (BH q=0.10)"] = {k: {"p": round(v, 4), "passes": bool(o)} for (k, v), o in zip(fam.items(), ok)}
    return R


if __name__ == "__main__":
    for k, v in main(sys.argv[1]).items():
        print("==", k)
        for kk, vv in v.items():
            print("  ", kk, json.dumps(vv))
