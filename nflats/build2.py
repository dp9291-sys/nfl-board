"""Build v2 feature table on top of v1 (python -m nflats.build2 2026)."""
import sys
import numpy as np
import pandas as pd
from . import data, advanced as A
from .ratings import TIDX

FIRST = 2013


def build2(last):
    v1 = pd.read_parquet(f"{data.DATA_DIR}/features.parquet")
    games = data.load_games()
    games = games[(games.season >= FIRST - 1) & (games.season <= last)]
    pbp = A.load_pbp2(range(max(2009, FIRST - 4), last + 1))
    tg = A.team_game_v2(pbp[pbp.season >= FIRST - 1])
    qg = A.qb_game_v2(pbp)
    rows = []
    targets = games[games.season >= FIRST][["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    for season, week in targets.itertuples(index=False):
        gw = games[(games.season == season) & (games.week == week)]
        r = A.ratings_v2_at(tg, season, week)
        cp = A.qb_cpoe_at(qg, season, week)
        mr, mhfa = A.market_ratings_at(games, season, week)
        for gm in gw.itertuples():
            h, a = TIDX[gm.home_team], TIDX[gm.away_team]
            row = {"game_id": gm.game_id}
            for k, v in r.items():
                row[f"h2_{k}"], row[f"a2_{k}"] = v[h], v[a]
            row["h_qb_cpoe"] = cp.get(gm.home_qb_id, -1.5) if isinstance(gm.home_qb_id, str) else -1.5
            row["a_qb_cpoe"] = cp.get(gm.away_qb_id, -1.5) if isinstance(gm.away_qb_id, str) else -1.5
            row["mkt_line_from_ratings"] = mr[h] - mr[a] + (mhfa if gm.location != "Neutral" else 0)
            rows.append(row)
        if week == 1:
            print("v2", season, flush=True)
    f = v1[v1.season >= FIRST].merge(pd.DataFrame(rows), on="game_id", how="inner")

    # team-baseline CPOE the ratings were built with (dropback-weighted), for QB-change adjustment
    prev = qg.copy()
    # (approximation: use team's season-to-date + last season CPOE as baseline via v2 off_cpoe)
    P = 62.0
    D = lambda k: (f[f"h2_off_{k}"] - f[f"h2_def_{k}"]) - (f[f"a2_off_{k}"] - f[f"a2_def_{k}"])
    f["early_diff"] = P * D("early")
    f["nto_diff"] = P * D("nto")
    f["expl_diff"] = 100 * D("expl")
    f["sack_diff"] = -100 * D("sack")          # positive = home protects QB better / gets more sacks
    f["pass_diff"] = 36 * D("pass")
    f["st_diff"] = f.h2_off_st - f.a2_off_st    # net ST EPA per game
    f["to_diff"] = -100 * ((f.h2_off_int_rate - f.h2_def_int_rate) - (f.a2_off_int_rate - f.a2_def_int_rate))
    f["fum_diff"] = -100 * ((f.h2_off_fum_rate - f.h2_def_fum_rate) - (f.a2_off_fum_rate - f.a2_def_fum_rate))
    f["proe_diff"] = 100 * (f.h2_off_proe - f.a2_off_proe)
    f["pen_diff"] = -(f.h2_off_pen - f.a2_off_pen) / 10
    # QB CPOE change vs what the team's passing ratings were built with
    f["qb_cpoe_diff"] = (f.h_qb_cpoe - f.h2_off_cpoe) - (f.a_qb_cpoe - f.a2_off_cpoe)

    # injuries
    inj, snaps, players = A.load_injury_inputs(range(FIRST, last + 1))
    inj_f, _ = A.injury_features(inj, snaps, players)
    for side, team in [("h", "home_team"), ("a", "away_team")]:
        x = inj_f.rename(columns={g: f"{side}_inj_{g}" for g in A.INJ_GROUPS}).rename(columns={"team": team})
        f = f.merge(x, on=["season", "week", team], how="left")
    for g in A.INJ_GROUPS:
        f[f"h_inj_{g}"] = f[f"h_inj_{g}"].fillna(0)
        f[f"a_inj_{g}"] = f[f"a_inj_{g}"].fillna(0)
        f[f"inj_{g}_diff"] = f[f"h_inj_{g}"] - f[f"a_inj_{g}"]   # positive = home missing more
    f["inj_off_diff"] = sum(f[f"inj_{g}_diff"] for g in ["WR", "TE", "RB", "OL"])
    f["inj_def_diff"] = sum(f[f"inj_{g}_diff"] for g in ["DL", "LB", "DB"])

    # weather (historical actuals from games.csv; forecasts injected at predict time)
    outdoor = ~f.roof.isin(["dome", "closed"])
    f["wind_mph"] = np.where(outdoor, f.wind.fillna(f.wind.median()), 0.0)
    f["temp_f"] = np.where(outdoor, f.temp.fillna(55), 68.0)
    f["cold_away_warm"] = (outdoor & (f.temp_f <= 40) & f.away_team.isin(A.DOME_OR_WARM) & ~f.home_team.isin(A.DOME_OR_WARM)).astype(float)
    f["cold_home_warm"] = (outdoor & (f.temp_f <= 40) & f.home_team.isin(A.DOME_OR_WARM) & ~f.away_team.isin(A.DOME_OR_WARM)).astype(float)
    f["wind15"] = (f.wind_mph >= 15).astype(float)

    # market features
    f["mkt_dev"] = f.spread_line - f.mkt_line_from_ratings
    la = A.last_ats(games)
    for side, team in [("h", "home_team"), ("a", "away_team")]:
        x = la[["team", "t", "season", "ats"]].rename(columns={"team": team, "ats": f"{side}_last_ats", "t": "t_l", "season": "s_l"})
        tmp = f[["game_id", team, "season", "week"]].assign(t=(f.season * 100 + f.week).astype("int64")).sort_values("t")
        mm = pd.merge_asof(tmp, x.sort_values("t_l"), left_on="t", right_on="t_l", by=team,
                           allow_exact_matches=False, direction="backward")
        mm.loc[mm.s_l != mm.season, f"{side}_last_ats"] = 0.0
        f = f.merge(mm[["game_id", f"{side}_last_ats"]], on="game_id", how="left")
    f["last_ats_diff"] = f.h_last_ats.fillna(0) - f.a_last_ats.fillna(0)
    return f


if __name__ == "__main__":
    last = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    f = build2(last)
    f.to_parquet(f"{data.DATA_DIR}/features2.parquet")
    print(f.shape)
