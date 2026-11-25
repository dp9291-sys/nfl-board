"""Luck-stripped special-teams rating (python -m nflats.stluck).

Pre-registered variants, each built only from games before the week being predicted:
  raw   : the board's rating: net EPA per game on kickoffs, punts and extra points (nflverse's special-teams
          flag leaves field goals out), decayed, K=4 shrink                       -> sanity check vs board
  all   : raw plus field goals at their actual result
  A     : raw with kicking luck removed (FG and XP at expected value given distance, wind, cold, altitude;
          the league curve is fitted on prior seasons only) and every play's EPA capped at the 1st/99th
          percentile of 1999-2001 special-teams plays (blocked kicks, return TDs, muffs count less)
  AF    : A plus field goals at expected value (attempts count, makes and misses don't)
  B     : A + punts adjusted for wind, cold, dome and altitude (fitted on the 5 prior seasons)
  C     : B, opponent-adjusted (ridge: net = rating(team) - rating(opponent), same decay, lambda 4)
  BF/CF : B/C plus field goals at expected value
Special-teams plays are identified by play type (kickoff, punt, field goal, extra point), since the
special_teams_play flag is missing before 2006.
Outputs data/stluck.parquet: game_id + home-minus-away differentials for each variant.
"""
import os
import numpy as np
import pandas as pd
from . import data
from .ratings import TEAMS, TIDX, NT, _weights, _ridge

FIRST = 1999
COLS = ["game_id", "season", "week", "posteam", "defteam", "home_team", "away_team", "play_type", "epa",
        "kick_distance", "field_goal_result", "extra_point_result", "temp", "wind", "roof", "season_type"]
KINDS = {"field_goal": "fg", "extra_point": "xp", "punt": "punt", "kickoff": "ko"}


def load(last):
    fr = []
    for y in range(FIRST, last + 1):
        p = pd.read_parquet(os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"), columns=COLS)
        fr.append(p[p.play_type.isin(KINDS) & p.epa.notna() & p.posteam.notna()])
    st = pd.concat(fr, ignore_index=True)
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        st[c] = st[c].replace(data.TEAM_MAP)
    st["kind"] = st.play_type.map(KINDS)
    indoor = st.roof.isin(["dome", "closed"])
    st["indoor"] = indoor.astype(float)
    st["wind_eff"] = np.where(indoor, 0.0, st.wind.fillna(6.0))
    st["cold"] = np.where(indoor, 0.0, np.clip(50 - st.temp.fillna(60), 0, None) / 10)
    st["alt"] = (st.home_team == "DEN").astype(float)
    return st


def _logit(X, y, iters=40, lam=1e-3):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - lam * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def _fgX(x):
    d = (x.kick_distance.values.astype(float) - 40) / 10
    w = x.wind_eff.values / 10
    return np.c_[np.ones_like(d), d, d * d, w, w * np.clip(d, 0, None), x.cold.values, x.alt.values]


def luck_strip(st):
    """Per-play EPA with kicking luck removed (A) and punts weather-adjusted (B), fitted season by season on prior seasons."""
    st = st.copy()
    st["make"] = np.where(st.kind == "fg", (st.field_goal_result == "made").astype(float),
                          np.where(st.kind == "xp", (st.extra_point_result == "good").astype(float), np.nan))
    st["e_A"] = st.epa.astype(float)
    for S in sorted(st.season.unique()):
        cur = st.season == S
        prior = st[(st.season < S) & (st.season >= S - 8)]
        if S == FIRST:  # no prior season: use the season itself (only feeds 1999 ratings, never a test season)
            prior = st[cur]
        # field goals: league make curve with weather, and the EPA value of a make by 5-yard distance bin
        fp = prior[(prior.kind == "fg") & prior.kick_distance.notna()]
        b = _logit(_fgX(fp), fp.make.values)
        f = cur & (st.kind == "fg") & st.kick_distance.notna()
        p = 1 / (1 + np.exp(-(_fgX(st[f]) @ b)))
        binp = (fp.kick_distance // 5).clip(3, 12)
        val = fp.groupby([binp, fp.make]).epa.mean().unstack()
        val = (val[1.0] - val[0.0]).reindex(range(3, 13)).interpolate().bfill().ffill()
        v = val.reindex((st.loc[f, "kick_distance"] // 5).clip(3, 12).values).values
        st.loc[f, "e_A"] = st.loc[f, "epa"].values - (st.loc[f, "make"].values - p) * v
        # extra points: league make rate and EPA value of a make, by era (the kick moved back in 2015)
        xp_prior = prior[(prior.kind == "xp") & ((prior.season >= 2015) == (S >= 2015))]
        if len(xp_prior) < 500:
            xp_prior = st[cur & (st.kind == "xp")]
        pxp = xp_prior.make.mean()
        vxp = xp_prior.groupby("make").epa.mean()
        vxp = float(vxp.get(1.0, 0) - vxp.get(0.0, -1))
        x = cur & (st.kind == "xp")
        st.loc[x, "e_A"] = st.loc[x, "epa"].values - (st.loc[x, "make"].values - pxp) * vxp
    lo, hi = np.percentile(st.loc[st.season <= 2001, "e_A"], [1, 99])
    st["e_A"] = st.e_A.clip(lo, hi)
    isfg = (st.kind == "fg").values
    st["e_AF"] = st.e_A                                   # with field goals at expected value
    st["cur"] = np.where(isfg, 0.0, st.epa)               # the board's rating: kickoffs, punts, XPs (no FGs)
    st["e_A"] = np.where(isfg, 0.0, st.e_A)               # A keeps the board's scope
    # B: punts relative to what that weather/venue usually produces
    st["e_B"] = st.e_A.copy()
    for S in sorted(st.season.unique()):
        cur = (st.season == S) & (st.kind == "punt")
        pr = st[(st.season < S) & (st.season >= S - 5) & (st.kind == "punt")]
        if len(pr) < 2000:
            continue
        X = np.c_[np.ones(len(pr)), pr.wind_eff / 10, pr.cold, pr.indoor, pr.alt]
        coef = np.linalg.lstsq(X, pr.e_A.values, rcond=None)[0]
        Xc = np.c_[np.ones(cur.sum()), st.loc[cur, "wind_eff"] / 10, st.loc[cur, "cold"], st.loc[cur, "indoor"], st.loc[cur, "alt"]]
        st.loc[cur, "e_B"] = st.loc[cur, "e_A"].values - (Xc @ coef - coef[0])
    st["e_BF"] = st.e_B + np.where(isfg, st.e_AF, 0.0)
    return st, (lo, hi)


def team_games(st, games):
    """Net (team minus opponent) per team-game for each per-play column."""
    out = []
    for col, name in [("epa", "all"), ("cur", "raw"), ("e_A", "A"), ("e_AF", "AF"), ("e_B", "B"), ("e_BF", "BF")]:
        a = st.groupby(["game_id", "posteam"])[col].sum()
        b = st.groupby(["game_id", "defteam"])[col].sum().rename_axis(["game_id", "posteam"])
        out.append(a.sub(b, fill_value=0).rename(name))
    net = pd.concat(out, axis=1).reset_index()
    g = games[["game_id", "season", "week", "home_team", "away_team"]]
    rows = pd.concat([g.assign(posteam=g.home_team, defteam=g.away_team, home=1),
                      g.assign(posteam=g.away_team, defteam=g.home_team, home=0)])
    return rows.merge(net, on=["game_id", "posteam"], how="inner")


def _decayed(d, w, col, K=4.0):
    s = pd.Series(w * d[col].values).groupby(d.posteam.values).sum()
    ws = pd.Series(w).groupby(d.posteam.values).sum()
    lg = float((w * d[col].values).sum() / w.sum())
    return ((s + K * lg) / (ws + K)).reindex(TEAMS).fillna(lg)


def _opp_adj(d, w, col, lam=4.0):
    n = len(d)
    X = np.zeros((n, NT))
    X[np.arange(n), d.posteam.map(TIDX).values] = 1.0
    X[np.arange(n), d.defteam.map(TIDX).values] -= 1.0
    b = _ridge(X, d[col].values, w, lam, [])
    return pd.Series(b, index=TEAMS)


def ratings(tg, games):
    rows = []
    sched = games.sort_values(["season", "week"])
    for (S, W), wk in sched.groupby(["season", "week"], sort=True):
        if S == FIRST:
            continue
        m = ((tg.season == S) & (tg.week < W)) | (tg.season == S - 1)
        d = tg[m]
        prev_max = tg.loc[tg.season == S - 1, "week"].max()
        w = _weights(d, S, W, prev_max)
        R = {k: _decayed(d, w, k) for k in ["all", "raw", "A", "AF", "B", "BF"]}
        R["C"] = _opp_adj(d, w, "B")
        R["CF"] = _opp_adj(d, w, "BF")
        for r in wk.itertuples():
            rec = {"game_id": r.game_id}
            for k, v in R.items():
                rec[f"sl_{k}"] = float(v[r.home_team] - v[r.away_team])
                rec[f"sl_{k}_h"], rec[f"sl_{k}_a"] = float(v[r.home_team]), float(v[r.away_team])
            rows.append(rec)
    return pd.DataFrame(rows)


def build(last):
    games = data.load_games()
    games = games[(games.season >= FIRST) & (games.season <= last) & games.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])]
    for c in ["home_team", "away_team"]:
        games[c] = games[c].replace(data.TEAM_MAP)
    st = load(last)
    st, caps = luck_strip(st)
    tg = team_games(st, games)
    r = ratings(tg, games)
    r.to_parquet(os.path.join(data.DATA_DIR, "stluck.parquet"))
    tg.to_parquet(os.path.join(data.DATA_DIR, "stluck_tg.parquet"))
    return r, caps


if __name__ == "__main__":
    import sys
    r, caps = build(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
    print(len(r), "games; play EPA caps", np.round(caps, 2))
