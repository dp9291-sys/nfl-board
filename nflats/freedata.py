"""Free-data signal tests (python -m nflats.freedata). Every feature uses only information from before the game.

1 ref_home     : referee's earlier games, home ATS (wins - losses) / (n + 50)            [1999-2026]
2 surf_unfam   : road team's home surface type (grass vs artificial) differs from this field [1999-2026]
3 q_dnp_diff   : Questionable players who did not practice (final report), snap share x position value,
                 home minus away                                                         [2012-2026]
4 press_diff   : expected pressure rate on home QB minus on away QB (PFR, decayed, shrunk) [2018-2026]
5 mtkl_diff    : home defense missed-tackle rate minus away (PFR, decayed, shrunk)         [2018-2026]
6 ol_cont_diff : core-5 OL (most snaps over prior 4 games) who played 50%+ in each of the last 3 games
                 and are not Out/Doubtful this week, home minus away                      [2013-2026]
"""
import glob
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from . import data, validate as V
from .ratings import TEAMS, _weights

POSVAL = {"WR": 1.3, "TE": 1.0, "RB": 0.6, "FB": 0.3, "LB": 0.8, "ILB": 0.8, "OLB": 0.8, "MLB": 0.8, "CB": 0.5, "S": 0.5,
          "SS": 0.5, "FS": 0.5, "DB": 0.5, "T": 0.4, "G": 0.4, "C": 0.4, "OL": 0.4, "OT": 0.4, "OG": 0.4,
          "DE": 0.4, "DT": 0.4, "DL": 0.4, "NT": 0.4}
OL = {"T", "G", "C", "OL", "OT", "OG"}
ART = {"fieldturf", "astroturf", "sportturf", "matrixturf", "astroplay", "a_turf", "dessograss"}


def games():
    g = data.load_games()
    g = g[g.game_type == "REG"].copy()
    g["surface"] = g.surface.astype(str).str.strip().str.lower()
    g["art"] = g.surface.isin(ART)
    return g.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def f_referee(g):
    out = {}
    hist = {}
    for (S, W), wk in g.groupby(["season", "week"], sort=True):
        for r in wk.itertuples():
            h = hist.get(r.referee)
            out[r.game_id] = 0.0 if (h is None or r.referee != r.referee) else (h[0] - h[1]) / (h[0] + h[1] + 50)
        for r in wk.itertuples():
            if r.referee == r.referee and r.result == r.result and r.spread_line == r.spread_line:
                a = r.result - r.spread_line
                h = hist.setdefault(r.referee, [0, 0])
                if a > 0:
                    h[0] += 1
                elif a < 0:
                    h[1] += 1
    return pd.Series(out, name="ref_home")


def f_surface(g):
    last_home = {}
    out = {}
    for (S, W), wk in g.groupby(["season", "week"], sort=True):
        for r in wk.itertuples():
            ah = last_home.get(r.away_team)
            out[r.game_id] = 0.0 if (ah is None or r.location != "Home" or r.surface in ("", "nan")) else float(ah != r.art)
        for r in wk.itertuples():
            if r.location == "Home" and r.surface not in ("", "nan"):
                last_home[r.home_team] = r.art
    return pd.Series(out, name="surf_unfam")


def _snaps():
    fr = []
    for p in sorted(glob.glob(os.path.join(data.DATA_DIR, "snap_*.parquet"))) + [os.path.join(data.DATA_DIR, "snap26.parquet")]:
        if os.path.exists(p):
            fr.append(pd.read_parquet(p, columns=["game_id", "season", "week", "game_type", "pfr_player_id", "position", "team",
                                                  "offense_pct", "defense_pct"]))
    s = pd.concat(fr, ignore_index=True).drop_duplicates(["game_id", "pfr_player_id"])
    s["team"] = s.team.replace(data.TEAM_MAP)
    s["pct"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["t"] = (s.season * 100 + s.week).astype("int64")
    return s[s.game_type == "REG"]


def _injuries():
    fr = []
    for y in range(2012, 2027):
        p = os.path.join(data.DATA_DIR, f"inj_{y}.parquet")
        if os.path.exists(p):
            fr.append(pd.read_parquet(p, columns=["season", "week", "team", "gsis_id", "position", "report_status", "practice_status"]))
    i = pd.concat(fr, ignore_index=True)
    i["team"] = i.team.replace(data.TEAM_MAP)
    return i


def f_practice(g, s, inj, players):
    s = s.sort_values(["pfr_player_id", "t"])
    s["roll"] = s.groupby("pfr_player_id").pct.transform(lambda x: x.rolling(6, min_periods=1).mean())
    q = inj[(inj.report_status == "Questionable") & inj.practice_status.astype(str).str.startswith("Did Not")].copy()
    q = q.merge(players, on="gsis_id", how="inner")
    q["t"] = (q.season * 100 + q.week).astype("int64")
    q = q.sort_values("t")
    m = pd.merge_asof(q, s[["pfr_player_id", "t", "roll"]].rename(columns={"pfr_player_id": "pfr_id", "t": "t_s"}).sort_values("t_s"),
                      left_on="t", right_on="t_s", by="pfr_id", allow_exact_matches=False, direction="backward")
    m = m[(m.t - m.t_s) < 150]
    m["val"] = m.roll * m.position.map(POSVAL).fillna(0.3)
    tv = m.groupby(["season", "week", "team"]).val.sum()
    hv = g.apply(lambda r: tv.get((r.season, r.week, r.home_team), 0.0), axis=1)
    av = g.apply(lambda r: tv.get((r.season, r.week, r.away_team), 0.0), axis=1)
    return pd.Series((hv - av).values, index=g.game_id.values, name="q_dnp_diff")


def f_ol(g, s, inj, players):
    ol = s[s.position.isin(OL)].copy()
    ol = ol.sort_values("t")
    out = {}
    outs = inj[inj.report_status.isin(["Out", "Doubtful"])].merge(players, on="gsis_id", how="inner")
    outset = set(zip(outs.season, outs.week, outs.team, outs.pfr_id))
    by_team = {t: x for t, x in ol.groupby("team")}
    for r in g[g.season >= 2013].itertuples():
        t0 = r.season * 100 + r.week
        vals = []
        for tm in (r.home_team, r.away_team):
            x = by_team.get(tm)
            if x is None:
                vals.append(np.nan)
                continue
            prev = x[x.t < t0]
            gids = prev.drop_duplicates("game_id").sort_values("t").game_id.tolist()
            if len(gids) < 4:
                vals.append(np.nan)
                continue
            last4 = prev[prev.game_id.isin(gids[-4:])]
            core = last4.groupby("pfr_player_id").pct.sum().nlargest(5).index
            last3 = prev[prev.game_id.isin(gids[-3:])]
            ok = 0
            for pid in core:
                pl = last3[last3.pfr_player_id == pid]
                if len(pl) == 3 and (pl.pct >= 0.5).all() and (r.season, r.week, tm, pid) not in outset:
                    ok += 1
            vals.append(ok)
        out[r.game_id] = vals[0] - vals[1] if not np.isnan(vals).any() else np.nan
    return pd.Series(out, name="ol_cont_diff")


def f_pfr(g):
    fr = []
    for y in range(2018, 2027):
        a = os.path.join(data.DATA_DIR, f"pfr_pass_{y}.parquet")
        b = os.path.join(data.DATA_DIR, f"pfr_def_{y}.parquet")
        if os.path.exists(a) and os.path.exists(b):
            pa = pd.read_parquet(a)
            de = pd.read_parquet(b)
            pa = pa[pa.game_type == "REG"].groupby(["game_id", "season", "week", "team", "opponent"]).agg(press=("times_pressured", "sum")).reset_index()
            de = de[de.game_type == "REG"].groupby(["game_id", "team"]).agg(mt=("def_missed_tackles", "sum"), tk=("def_tackles_combined", "sum")).reset_index()
            fr.append(pa.merge(de, on=["game_id", "team"], how="left"))
    p = pd.concat(fr, ignore_index=True)
    for c in ["team", "opponent"]:
        p[c] = p[c].replace(data.TEAM_MAP)
    # dropbacks from pbp
    db = []
    for y in range(2018, 2027):
        x = pd.read_parquet(os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"), columns=["game_id", "posteam", "qb_dropback"])
        db.append(x[x.qb_dropback == 1].groupby(["game_id", "posteam"]).size().rename("db").reset_index())
    db = pd.concat(db)
    db["posteam"] = db.posteam.replace(data.TEAM_MAP)
    p = p.merge(db.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="inner")
    out_p, out_m = {}, {}
    for (S, W), wk in g[g.season >= 2018].groupby(["season", "week"], sort=True):
        m = ((p.season == S) & (p.week < W)) | (p.season == S - 1)
        d = p[m]
        if not len(d):
            continue
        prev_max = p.loc[p.season == S - 1, "week"].max()
        w = _weights(d, S, W, prev_max if prev_max == prev_max else 18)
        L = np.sum(w * d.press) / np.sum(w * d.db)
        K = 4.0 * float(np.average(d.db, weights=w))
        so = pd.Series(w * d.press.values).groupby(d.team.values).sum()
        no = pd.Series(w * d.db.values).groupby(d.team.values).sum()
        off = ((so + K * L) / (no + K)).to_dict()            # pressure allowed by offense
        sd = pd.Series(w * d.press.values).groupby(d.opponent.values).sum()
        nd = pd.Series(w * d.db.values).groupby(d.opponent.values).sum()
        dfn = ((sd + K * L) / (nd + K)).to_dict()            # pressure generated by defense
        Lm = np.sum(w * d.mt) / np.sum(w * (d.tk + d.mt))
        km = pd.Series(w * d.mt.values).groupby(d.team.values).sum()
        kn = pd.Series(w * (d.tk + d.mt).values).groupby(d.team.values).sum()
        mt = ((km + 60 * Lm) / (kn + 60)).to_dict()
        for r in wk.itertuples():
            eh = off.get(r.home_team, L) + dfn.get(r.away_team, L) - L
            ea = off.get(r.away_team, L) + dfn.get(r.home_team, L) - L
            out_p[r.game_id] = eh - ea
            out_m[r.game_id] = mt.get(r.home_team, Lm) - mt.get(r.away_team, Lm)
    return pd.Series(out_p, name="press_diff"), pd.Series(out_m, name="mtkl_diff")


def build():
    g = games()
    players = data.load_players()[["gsis_id", "pfr_id"]].dropna() if hasattr(data, "load_players") else \
        pd.read_parquet(os.path.join(data.DATA_DIR, "players.parquet"), columns=["gsis_id", "pfr_id"]).dropna()
    s = _snaps()
    inj = _injuries()
    feats = [f_referee(g), f_surface(g)]
    print("  referee, surface done", flush=True)
    feats.append(f_practice(g[g.season >= 2012], s, inj, players))
    print("  practice done", flush=True)
    feats += list(f_pfr(g))
    print("  pfr done", flush=True)
    feats.append(f_ol(g, s, inj, players))
    print("  ol continuity done", flush=True)
    F = pd.concat(feats, axis=1)
    F.index.name = "game_id"
    F = F.reset_index()
    F.to_parquet(os.path.join(data.DATA_DIR, "freedata.parquet"))
    return g, F


def test(g, F):
    x = g.merge(F, on="game_id", how="left")
    x = x[x.result.notna() & x.spread_line.notna()].reset_index(drop=True)
    x["ats"] = (x.result - x.spread_line).clip(-21, 21)
    x["ats_raw"] = x.result - x.spread_line
    out, pv = {}, {}
    spec = {"ref_home": (range(2004, 2027), 1999), "surf_unfam": (range(2004, 2027), 1999), "q_dnp_diff": (range(2016, 2027), 2013),
            "press_diff": (range(2020, 2027), 2018), "mtkl_diff": (range(2020, 2027), 2018), "ol_cont_diff": (range(2016, 2027), 2013)}
    for c, (T, first) in spec.items():
        T = list(T)
        y = x[x[c].notna()].reset_index(drop=True)
        adj = V.oos_adjust(y, [c], T, first=first)
        m = y.season.isin(T)
        sc = V.score(y[m], adj[m])
        pv[c] = sc["p"]
        l = m & (adj.abs() >= 0.5) & (y.ats_raw != 0)
        out[c] = {"seasons": f"{T[0]}-{T[-1]}", "t": sc["t"], "p": round(sc["p"], 4), "top_third_win": sc["top_hit"],
                  "lean_0.5": f"{100 * float((np.sign(adj[l]) == np.sign(y.ats_raw[l])).mean()):.1f}% n{int(l.sum())}" if l.sum() else "none"}
    # direct rules with an independent pre-2015 check (no fitting)
    for lab, lo, hi in [("1999-2014", 1999, 2014), ("2015-2026", 2015, 2026)]:
        y = x[x.season.between(lo, hi) & (x.surf_unfam == 1) & (x.ats_raw != 0)]
        out[f"rule: home vs road team on unfamiliar surface, {lab}"] = f"{100 * float((y.ats_raw > 0).mean()):.1f}% home covers, n{len(y)}"
        y = x[x.season.between(lo, hi) & (x.ref_home.abs() >= 0.04) & (x.ats_raw != 0)]
        out[f"rule: follow referee's home tendency (|dev| >= 0.04), {lab}"] = f"{100 * float((np.sign(y.ref_home) == np.sign(y.ats_raw)).mean()):.1f}%, n{len(y)}"
    keys = list(pv)
    ok = V.bh([pv[k] for k in keys], q=0.10)
    out["FDR (BH q=0.10) passes"] = [k for k, o in zip(keys, ok) if o]
    return out


if __name__ == "__main__":
    import json
    g, F = build()
    for k, v in test(g, F).items():
        print(k, json.dumps(v))
