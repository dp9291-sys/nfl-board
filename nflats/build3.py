"""v3 candidate ATS signals on top of v2 features (python -m nflats.build3 2026).

New, all computed from information available before kickoff:
  - Special teams split: field goals/XPs, punts, kickoffs (net EPA per game, decayed)
  - Injury clusters: 2+ starters out in OL / secondary / receivers
  - Style matchups: pass offense x opposing pass defense, sack-prone OL x pass rush
  - Coaching: new head coach (early season), 4th-down aggressiveness
  - Totals: market total vs stats-implied total, spread-to-total ratio, wind with dog/favourite
  - Market: closing line vs the market's own prior ratings (proxy for the week's line move)
"""
import sys
import numpy as np
import pandas as pd
from . import data, advanced as A
from .ratings import TEAMS, TIDX, NT, _weights, _ridge

FIRST = 2013


def st_split(pbp):
    st = pbp[(pbp.special_teams_play == 1) & pbp.epa.notna() & pbp.posteam.notna()]
    st = st.assign(kind=np.select([st.play_type.isin(["field_goal", "extra_point"]), st.play_type == "punt",
                                   st.play_type == "kickoff"], ["fg", "punt", "ko"], "other"))
    a = st.groupby(["game_id", "posteam", "kind"]).epa.sum()
    b = st.groupby(["game_id", "defteam", "kind"]).epa.sum().rename_axis(["game_id", "posteam", "kind"])
    net = a.sub(b, fill_value=0).unstack("kind", fill_value=0).reset_index()
    for k in ["fg", "punt", "ko"]:
        if k not in net:
            net[k] = 0.0
    return net[["game_id", "posteam", "fg", "punt", "ko"]]


def fourth_down(pbp):
    p = pbp[(pbp.down == 4) & pbp.ydstogo.le(3) & pbp.yardline_100.between(1, 60) & pbp.wp.between(0.1, 0.9)
            & pbp.posteam.notna() & pbp.play_type.isin(["pass", "run", "punt", "field_goal"])]
    p = p.assign(go=p.play_type.isin(["pass", "run"]).astype(int), n=1)
    return p.groupby(["game_id", "posteam"])[["go", "n"]].sum().reset_index()


def decayed_team_avg(tg, cols, season, week, K=4.0, per_game=True):
    prev_max = tg.loc[tg.season == season - 1, "week"].max()
    m = ((tg.season == season) & (tg.week < week)) | (tg.season == season - 1)
    d = tg[m].assign(_w=_weights(tg[m], season, week, prev_max))
    out = {}
    for c in cols:
        s = (d._w * d[c]).groupby(d.posteam).sum()
        wsum = d._w.groupby(d.posteam).sum()
        lg = (d._w * d[c]).sum() / d._w.sum()
        out[c] = ((s + K * lg) / (wsum + K)).reindex(TEAMS).values
    return out


def scoring_ratings(games, season, week, lam=4.0):
    gm = games[(((games.season == season) & (games.week < week)) | (games.season == season - 1)) & games.result.notna()]
    prev_max = games.loc[games.season == season - 1, "week"].max()
    w = _weights(gm, season, week, prev_max)
    rows_X, rows_y, rows_w = [], [], []
    n = len(gm)
    X = np.zeros((2 * n, 2 + 2 * NT))
    X[:, 0] = 1
    hi, ai = gm.home_team.map(TIDX).values, gm.away_team.map(TIDX).values
    neutral = (gm.location == "Neutral").values
    X[:n, 1] = np.where(neutral, 0, 1)
    X[n:, 1] = np.where(neutral, 0, -1)
    X[np.arange(n), 2 + hi] = 1
    X[np.arange(n), 2 + NT + ai] = 1
    X[n + np.arange(n), 2 + ai] = 1
    X[n + np.arange(n), 2 + NT + hi] = 1
    y = np.r_[gm.home_score.values, gm.away_score.values]
    b = _ridge(X, y, np.r_[w, w], lam, [0, 1])
    return b[0], b[1], b[2:2 + NT], b[2 + NT:]


def build3(last):
    f = pd.read_parquet(f"{data.DATA_DIR}/features2.parquet")
    games = data.load_games()
    games = games[(games.season >= FIRST - 1) & (games.season <= last)]
    cols = ["game_id", "season", "week", "posteam", "defteam", "home_team", "away_team", "epa", "play_type",
            "special_teams_play", "down", "ydstogo", "yardline_100", "wp"]
    import os
    pbp = pd.concat([pd.read_parquet(os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"), columns=cols)
                     for y in range(FIRST - 1, last + 1)], ignore_index=True)
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        pbp[c] = pbp[c].replace(data.TEAM_MAP)
    keys = pbp.groupby(["game_id", "posteam"]).agg(season=("season", "first"), week=("week", "first")).reset_index()
    tg = keys.merge(st_split(pbp), on=["game_id", "posteam"], how="left").merge(fourth_down(pbp), on=["game_id", "posteam"], how="left")
    tg = tg.fillna({"fg": 0, "punt": 0, "ko": 0, "go": 0, "n": 0})

    rows = []
    for season, week in f[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).itertuples(index=False):
        r = decayed_team_avg(tg, ["fg", "punt", "ko"], season, week)
        # 4th-down go rate: pooled decayed counts, shrunk to league with 15 situations
        prev_max = tg.loc[tg.season == season - 1, "week"].max()
        m = ((tg.season == season) & (tg.week < week)) | (tg.season == season - 1)
        d = tg[m].assign(_w=_weights(tg[m], season, week, prev_max))
        go = (d._w * d.go).groupby(d.posteam).sum()
        nn = (d._w * d.n).groupby(d.posteam).sum()
        lg = go.sum() / nn.sum()
        gr = ((go + 15 * lg) / (nn + 15)).reindex(TEAMS).values
        mu, hfa, O, Dd = scoring_ratings(games, season, week)
        for g in f[(f.season == season) & (f.week == week)].itertuples():
            h, a = TIDX[g.home_team], TIDX[g.away_team]
            hf = hfa if g.location != "Neutral" else 0
            rows.append({"game_id": g.game_id,
                         "st_fg_diff": r["fg"][h] - r["fg"][a], "st_punt_diff": r["punt"][h] - r["punt"][a],
                         "st_ko_diff": r["ko"][h] - r["ko"][a],
                         "go_rate_diff": 100 * (gr[h] - gr[a]),
                         "model_total": 2 * mu + O[h] - Dd[a] + O[a] - Dd[h],
                         "model_margin_pts": (O[h] - Dd[a] + hf) - (O[a] - Dd[h] - hf)})
        if week == 1:
            print("v3", season, flush=True)
    x = f.merge(pd.DataFrame(rows), on="game_id", how="left")

    # injury clusters (2+ meaningful starters out at one position group)
    inj, snaps, players = A.load_injury_inputs(range(FIRST, last + 1))
    _, m = A.injury_features(inj, snaps, players)
    m = m[(m.report_status.isin(["Out", "Doubtful"])) & (m.roll >= 0.6)]
    m["grp2"] = m.grp.map({"OL": "ol", "DB": "db", "WR": "rec", "TE": "rec"})
    cl = m.dropna(subset=["grp2"]).groupby(["season", "week", "team", "grp2"]).size().unstack(fill_value=0)
    cl = (cl >= 2).astype(float).reindex(columns=["ol", "db", "rec"], fill_value=0).reset_index()
    for side, team in [("h", "home_team"), ("a", "away_team")]:
        x = x.merge(cl.rename(columns={"team": team, "ol": f"{side}_cl_ol", "db": f"{side}_cl_db", "rec": f"{side}_cl_rec"}),
                    on=["season", "week", team], how="left")
    for k in ["ol", "db", "rec"]:
        x[f"cl_{k}_diff"] = x[f"h_cl_{k}"].fillna(0) - x[f"a_cl_{k}"].fillna(0)

    # style matchups (products of opponent-adjusted unit ratings)
    x["pass_match"] = 100 * (x.h2_off_pass * x.a2_def_pass - x.a2_off_pass * x.h2_def_pass)
    x["sack_match"] = 1e4 * (x.h2_off_sack * x.a2_def_sack - x.a2_off_sack * x.h2_def_sack)

    # coaching: new head coach this season (vs team's coach last season), weeks 1-6
    gc = data.load_games()
    hc = pd.concat([gc[["season", "home_team", "home_coach"]].rename(columns={"home_team": "team", "home_coach": "coach"}),
                    gc[["season", "away_team", "away_coach"]].rename(columns={"away_team": "team", "away_coach": "coach"})])
    first_coach = hc.groupby(["team", "season"]).coach.agg(lambda s: s.mode().iloc[0]).reset_index()
    first_coach["prev"] = first_coach.groupby("team").coach.shift(1)
    first_coach["new_hc"] = (first_coach.coach != first_coach.prev) & first_coach.prev.notna()
    for side, team in [("h", "home_team"), ("a", "away_team")]:
        x = x.merge(first_coach[["team", "season", "new_hc"]].rename(columns={"team": team, "new_hc": f"{side}_new_hc"}),
                    on=["season", team], how="left")
    early = (x.week <= 6).astype(float)
    x["new_hc_early_diff"] = early * (x.h_new_hc.fillna(False).astype(float) - x.a_new_hc.fillna(False).astype(float))

    # totals-based indicators
    x["total_dev"] = x.total_line - x.model_total              # market expects more (+) / less (-) scoring than stats
    fav_sign = np.sign(x.spread_line).replace(0, 1)              # +1 home favoured
    x["fav_low_total"] = fav_sign * (np.abs(x.spread_line) / x.total_line.clip(lower=30))  # big spread relative to total
    x["total_dev_x_fav"] = fav_sign * x.total_dev                # does a higher-than-expected total help the favourite?
    x["wind_x_fav"] = fav_sign * (x.wind_mph >= 10).astype(float)
    x["model_line_gap_pts"] = x.model_margin_pts - x.spread_line  # points-only scoring model vs close
    return x


if __name__ == "__main__":
    last = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    x = build3(last)
    x.to_parquet(f"{data.DATA_DIR}/features3.parquet")
    print(x.shape)
