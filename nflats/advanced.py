"""v2 data: secondary/tertiary efficiency stats, special teams, QB CPOE, injuries, weather, market features.

Everything is computed so a value for (season, week) uses only information available before that week's games.
"""
import os
import numpy as np
import pandas as pd
from .data import DATA_DIR, TEAM_MAP
from .ratings import TEAMS, TIDX, NT, _weights, _ridge
from . import ratings

PBP_COLS2 = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
             "epa", "success", "qb_dropback", "qb_epa", "wp", "play_type", "down", "yards_gained",
             "interception", "fumble_lost", "fumble", "sack", "qb_hit", "cpoe", "pass_oe",
             "special_teams_play", "penalty_team", "penalty_yards", "passer_player_id",
             "passer_player_name"]

DOME_OR_WARM = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "LA", "LAC", "MIN", "NO", "MIA", "TB", "JAX"}


def load_pbp2(seasons):
    fr = []
    for y in seasons:
        fr.append(pd.read_parquet(os.path.join(DATA_DIR, f"pbp_{y}.parquet"), columns=PBP_COLS2))
    p = pd.concat(fr, ignore_index=True)
    for c in ["posteam", "defteam", "home_team", "away_team", "penalty_team"]:
        p[c] = p[c].replace(TEAM_MAP)
    return p


def team_game_v2(pbp):
    """Per (game, offense) row with secondary stats. Scrimmage plays, WP 5-95%."""
    p = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna() & pbp.posteam.notna()]
    p = p[(p.wp >= 0.05) & (p.wp <= 0.95)].copy()
    db = p.qb_dropback.fillna(0).astype(int)
    to = ((p.interception.fillna(0) == 1) | (p.fumble_lost.fillna(0) == 1)).astype(int)
    early = p.down.isin([1, 2]).astype(int)
    expl = (((db == 1) & (p.yards_gained >= 20)) | ((db == 0) & (p.yards_gained >= 10))).astype(int)
    neutral = early.astype(bool) & p.wp.between(0.2, 0.8) & p.pass_oe.notna()
    p = p.assign(db=db, pass_epa=p.epa * db, early=early, early_epa=p.epa * early,
                 nto=1 - to, nto_epa=p.epa * (1 - to), expl=expl,
                 sk=p.sack.fillna(0) * db, hit=p.qb_hit.fillna(0) * db,
                 ints=p.interception.fillna(0), fum=p.fumble.fillna(0),
                 cp=p.cpoe.fillna(0), cpn=p.cpoe.notna().astype(int),
                 proe=np.where(neutral, p.pass_oe, 0.0), proen=neutral.astype(int))
    keys = ["game_id", "season", "week", "posteam", "defteam", "home_team"]
    agg = p.groupby(keys)[["epa", "success", "db", "pass_epa", "early", "early_epa", "nto", "nto_epa",
                           "expl", "sk", "hit", "ints", "fum", "cp", "cpn", "proe", "proen"]].sum().reset_index()
    agg["plays"] = p.groupby(keys).size().values
    agg["home"] = (agg.posteam == agg.home_team).astype(int)
    # special teams: net EPA for the team on all ST plays (no WP filter)
    st = pbp[(pbp.special_teams_play == 1) & pbp.epa.notna() & pbp.posteam.notna()]
    a = st.groupby(["game_id", "posteam"]).epa.sum()
    b = st.groupby(["game_id", "defteam"]).epa.sum()
    stn = a.sub(b.rename_axis(["game_id", "posteam"]), fill_value=0).rename("st_epa").reset_index()
    agg = agg.merge(stn, on=["game_id", "posteam"], how="left")
    pen = pbp[pbp.penalty_team.notna()].groupby(["game_id", "penalty_team"]).penalty_yards.sum()
    pen = pen.rename("pen_yds").rename_axis(["game_id", "posteam"]).reset_index()
    agg = agg.merge(pen, on=["game_id", "posteam"], how="left").fillna({"st_epa": 0, "pen_yds": 0})
    return agg


def qb_game_v2(pbp):
    p = pbp[(pbp.qb_dropback == 1) & pbp.qb_epa.notna() & pbp.passer_player_id.notna()]
    p = p[(p.wp >= 0.05) & (p.wp <= 0.95)]
    p = p.assign(cp=p.cpoe.fillna(0), cpn=p.cpoe.notna().astype(int))
    return p.groupby(["game_id", "season", "week", "posteam", "passer_player_id"]).agg(
        n=("qb_epa", "size"), epa=("qb_epa", "sum"), cp=("cp", "sum"), cpn=("cpn", "sum"),
        name=("passer_player_name", "last")).reset_index()


ADJ_METRICS = {  # metric name -> (numerator col, denominator col, ridge multiplier)
    "epa": ("epa", "plays", 1), "sr": ("success", "plays", 1), "early": ("early_epa", "early", 1.5),
    "nto": ("nto_epa", "nto", 1), "expl": ("expl", "plays", 2), "sack": ("sk", "db", 2),
    "pass": ("pass_epa", "db", 1),
}
RAW_METRICS = {  # decayed averages, not opponent adjusted (per game or per play)
    "st": ("st_epa", None), "int_rate": ("ints", "db"), "fum_rate": ("fum", "plays"),
    "proe": ("proe", "proen"), "pen": ("pen_yds", None), "cpoe": ("cp", "cpn"),
}


def ratings_v2_at(tg, season, week):
    prev_max = tg.loc[tg.season == season - 1, "week"].max()
    m = ((tg.season == season) & (tg.week < week)) | (tg.season == season - 1)
    d = tg[m]
    w = _weights(d, season, week, prev_max)
    n = len(d)
    X = np.zeros((n, 2 + 2 * NT))
    X[:, 0] = 1
    X[:, 1] = d.home.values * 2 - 1
    oi, di = d.posteam.map(TIDX).values, d.defteam.map(TIDX).values
    X[np.arange(n), 2 + oi] = 1
    X[np.arange(n), 2 + NT + di] = 1
    out = {}
    for k, (num, den, mult) in ADJ_METRICS.items():
        y = (d[num] / d[den].clip(lower=1)).values
        b = _ridge(X, y, w, ratings.LAM_EFF * mult, [0, 1])
        out[f"off_{k}"], out[f"def_{k}"] = b[2:2 + NT], b[2 + NT:]
    # raw decayed averages, shrunk to league mean with K game-equivalents
    K = 4.0
    for k, (num, den) in RAW_METRICS.items():
        for side, col in [("off", "posteam"), ("def", "defteam")]:
            if k in ("st", "pen") and side == "def":
                continue
            g = d.assign(_w=w)
            if den is None:
                s = g.groupby(col).apply(lambda x: (x._w * x[num]).sum(), include_groups=False)
                c = g.groupby(col)._w.sum()
                lg = (g._w * g[num]).sum() / g._w.sum()
                v = (s + K * lg) / (c + K)
            else:
                s = g.groupby(col).apply(lambda x: (x._w * x[num]).sum(), include_groups=False)
                c = g.groupby(col).apply(lambda x: (x._w * x[den]).sum(), include_groups=False)
                lg = (g._w * g[num]).sum() / (g._w * g[den]).sum()
                per = c / g.groupby(col)._w.sum()  # avg denominator per game
                v = (s + K * per * lg) / (c + K * per)
            out[f"{side}_{k}"] = np.array([v.get(t, np.nan) for t in TEAMS])
    return out


def qb_cpoe_at(qg, season, week, k=250.0, prior=-1.5):
    m = ((qg.season < season) | ((qg.season == season) & (qg.week < week))) & (qg.season >= season - 4)
    d = qg[m]
    age = (season - d.season) * 21 + (week - d.week)
    w = 0.5 ** (age / ratings.QB_HALF_LIFE)
    g = d.assign(wc=d.cp * w, wn=d.cpn * w).groupby("passer_player_id")[["wc", "wn"]].sum()
    return (g.wc + k * prior) / (g.wn + k)


# ---------------- injuries ----------------
POS_GROUP = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "HB": "RB",
             "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL", "LT": "OL", "RT": "OL", "LG": "OL", "RG": "OL",
             "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
             "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
             "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "SAF": "DB"}
INJ_GROUPS = ["WR", "TE", "RB", "OL", "DL", "LB", "DB"]
STATUS_W = {"Out": 1.0, "Doubtful": 0.9, "Questionable": 0.15}


def load_injury_inputs(seasons):
    inj = pd.concat([pd.read_parquet(os.path.join(DATA_DIR, f"inj_{y}.parquet")) for y in seasons], ignore_index=True)
    snaps = pd.concat([pd.read_parquet(os.path.join(DATA_DIR, f"snap_{y}.parquet")) for y in seasons
                       if y >= 2013], ignore_index=True)
    players = pd.read_parquet(os.path.join(DATA_DIR, "players.parquet"), columns=["gsis_id", "pfr_id"])
    for df_, c in [(inj, "team"), (snaps, "team")]:
        df_[c] = df_[c].replace(TEAM_MAP)
    return inj, snaps, players.dropna()


def injury_features(inj, snaps, players):
    """Snap-share-weighted absences by position group, per (season, week, team). Lookahead-free:
    snap shares come only from games before the report's week."""
    s = snaps.copy()
    s["pct"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["t"] = (s.season * 100 + s.week).astype("int64")
    s = s.sort_values(["pfr_player_id", "t"])
    s["roll"] = s.groupby("pfr_player_id").pct.transform(lambda x: x.rolling(6, min_periods=1).mean())
    i = inj[inj.report_status.isin(STATUS_W.keys())].merge(players, on="gsis_id", how="inner")
    i = i[i.position != "QB"].copy()
    i["t"] = (i.season * 100 + i.week).astype("int64")
    i["grp"] = i.position.map(POS_GROUP)
    i = i[i.grp.notna()].sort_values("t")
    m = pd.merge_asof(i, s[["pfr_player_id", "t", "roll"]].rename(columns={"pfr_player_id": "pfr_id", "t": "t_s"}).sort_values("t_s"),
                      left_on="t", right_on="t_s", by="pfr_id", allow_exact_matches=False, direction="backward")
    m = m[(m.t - m.t_s) < 150]  # snap history from this season or last season only
    m["val"] = m.roll * m.report_status.map(STATUS_W)
    f = m.pivot_table(index=["season", "week", "team"], columns="grp", values="val", aggfunc="sum", fill_value=0)
    f = f.reindex(columns=INJ_GROUPS, fill_value=0).reset_index()
    return f, m


# ---------------- market features ----------------
def market_ratings_at(games, season, week, lam=3.0):
    """Team ratings implied by prior closing spreads (decayed), and the market's own HFA."""
    gm = games[(((games.season == season) & (games.week < week)) | (games.season == season - 1))
               & games.spread_line.notna()]
    prev_max = games.loc[games.season == season - 1, "week"].max()
    w = _weights(gm, season, week, prev_max)
    n = len(gm)
    X = np.zeros((n, 1 + NT))
    X[:, 0] = (gm.location != "Neutral").astype(float).values
    X[np.arange(n), 1 + gm.home_team.map(TIDX).values] = 1
    X[np.arange(n), 1 + gm.away_team.map(TIDX).values] = -1
    b = _ridge(X, gm.spread_line.values, w, lam, [0])
    return b[1:], b[0]


def last_ats(games):
    """Each team's ATS margin in its previous game (capped), keyed by game_id for home/away."""
    g = games[games.result.notna() & games.spread_line.notna()].sort_values(["season", "week"])
    rows = []
    for r in g.itertuples():
        a = max(-21, min(21, r.result - r.spread_line))
        rows += [(r.season, r.week, r.home_team, a), (r.season, r.week, r.away_team, -a)]
    t = pd.DataFrame(rows, columns=["season", "week", "team", "ats"]).sort_values(["season", "week"])
    t["prev_ats"] = t.groupby(["team", "season"]).ats.shift(1)
    t["t"] = (t.season * 100 + t.week).astype("int64")
    return t
