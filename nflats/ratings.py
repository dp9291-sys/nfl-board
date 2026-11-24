"""Walk-forward team and QB ratings.

For each target (season, week) only games played BEFORE that week are used, so every
rating is something you could have known at bet time.
"""
import numpy as np
import pandas as pd

TEAMS = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB',
         'HOU', 'IND', 'JAX', 'KC', 'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ',
         'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS']
TIDX = {t: i for i, t in enumerate(TEAMS)}
NT = len(TEAMS)

# Tunables: set from common practice up front, NOT tuned on the backtest (avoids overfitting it)
HALF_LIFE = 9.0        # weeks
OFFSEASON_MULT = 0.55  # extra discount on last season's games
LAM_EFF = 4.0          # ridge penalty, in game-equivalents
LAM_MARGIN = 5.0
QB_HALF_LIFE = 34.0    # QBs are stickier than teams
QB_K = 300.0           # dropbacks of prior shrinkage
QB_PRIOR_GAP = 0.10    # unknown QB prior = league avg - this


def _weights(df, season, week, max_prev_week):
    cur = df["season"] == season
    age = np.where(cur, week - df["week"],
                   (week - 1) + (max_prev_week - df["week"] + 1))
    w = 0.5 ** (age / HALF_LIFE)
    w = np.where(cur, w, w * OFFSEASON_MULT)
    return w


def _ridge(X, y, w, lam, unpen):
    W = w[:, None]
    A = X.T @ (X * W)
    pen = np.full(X.shape[1], lam)
    pen[unpen] = 1e-6
    A += np.diag(pen)
    b = np.linalg.solve(A, X.T @ (w * y))
    return b


def team_ratings_at(tg, games, season, week):
    """tg: team_game_stats frame. games: game results frame. Returns per-team ratings dict."""
    prev_max = tg.loc[tg.season == season - 1, "week"].max()
    m = ((tg.season == season) & (tg.week < week)) | (tg.season == season - 1)
    d = tg[m]
    w = _weights(d, season, week, prev_max)
    n = len(d)
    X = np.zeros((n, 2 + 2 * NT))
    X[:, 0] = 1.0
    X[:, 1] = d["home"].values * 2 - 1  # +1 home offense, -1 away offense
    oi = d["posteam"].map(TIDX).values
    di = d["defteam"].map(TIDX).values
    X[np.arange(n), 2 + oi] = 1.0
    X[np.arange(n), 2 + NT + di] = 1.0
    out = {}
    for name, y in [("epa", (d["epa"] / d["plays"]).values),
                    ("sr", (d["succ"] / d["plays"]).values),
                    ("pass", (d["pass_epa"] / d["db"].clip(lower=1)).values),
                    ("rush", ((d["epa"] - d["pass_epa"]) / (d["plays"] - d["db"]).clip(lower=1)).values)]:
        lam = LAM_EFF * (4.0 if name == "rush" else 1.0)
        b = _ridge(X, y, w, lam, [0, 1])
        out[f"off_{name}"] = b[2:2 + NT]
        out[f"def_{name}"] = b[2 + NT:]  # positive = allows more (bad)

    # Point-margin rating (SRS style), one row per game
    gm = games[(((games.season == season) & (games.week < week)) | (games.season == season - 1))
               & games.result.notna()]
    w2 = _weights(gm, season, week, prev_max)
    n2 = len(gm)
    X2 = np.zeros((n2, 1 + NT))
    X2[:, 0] = (gm["location"] != "Neutral").astype(float).values
    X2[np.arange(n2), 1 + gm["home_team"].map(TIDX).values] = 1.0
    X2[np.arange(n2), 1 + gm["away_team"].map(TIDX).values] = -1.0
    y2 = gm["result"].clip(-28, 28).values  # cap blowouts
    b2 = _ridge(X2, y2, w2, LAM_MARGIN, [0])
    out["margin"] = b2[1:]
    out["hfa_est"] = b2[0]
    return out


def qb_values_at(qg, season, week):
    """Shrunk EPA/dropback for every QB, using games strictly before (season, week)."""
    m = (qg.season < season) | ((qg.season == season) & (qg.week < week))
    m &= qg.season >= season - 4
    d = qg[m]
    age = (season - d["season"]) * 21 + (week - d["week"])
    w = 0.5 ** (age / QB_HALF_LIFE)
    league = (d["epa"] * w).sum() / (d["n"] * w).sum()
    prior = league - QB_PRIOR_GAP
    grp = d.assign(we=d["epa"] * w, wn=d["n"] * w).groupby("passer_player_id")
    s = grp[["we", "wn"]].sum()
    val = (s["we"] + QB_K * prior) / (s["wn"] + QB_K)
    names = grp["name"].last()
    return val, prior, names, league


def team_qb_baseline(qg, qbval, prior, season, week):
    """Dropback-weighted QB value the team's current ratings were 'built with'."""
    prev_max = qg.loc[qg.season == season - 1, "week"].max()
    m = ((qg.season == season) & (qg.week < week)) | (qg.season == season - 1)
    d = qg[m].copy()
    d["w"] = _weights(d, season, week, prev_max) * d["n"]
    d["v"] = d["passer_player_id"].map(qbval).fillna(prior)
    b = d.groupby("posteam").apply(lambda x: (x["v"] * x["w"]).sum() / x["w"].sum(), include_groups=False)
    return b
