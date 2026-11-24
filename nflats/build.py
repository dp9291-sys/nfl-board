"""Build the walk-forward feature table: one row per game, features known before kickoff."""
import numpy as np
import pandas as pd
from . import data, ratings, features
from .ratings import TIDX

FIRST = 2010


def build(last_season, seasons_pbp=None, verbose=True):
    seasons_pbp = seasons_pbp or list(range(FIRST - 1, last_season + 1))
    games = data.load_games()
    games = games[(games.season >= FIRST - 1) & (games.season <= last_season)].copy()
    pbp = data.load_pbp(seasons_pbp)
    tg = data.team_game_stats(pbp)
    qg = data.qb_game_stats(pbp)
    rows = []
    targets = games[games.season >= FIRST][["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    for season, week in targets.itertuples(index=False):
        gw = games[(games.season == season) & (games.week == week)]
        if gw.empty:
            continue
        r = ratings.team_ratings_at(tg, games, season, week)
        qbv, prior, names, league = ratings.qb_values_at(qg, season, week)
        base = ratings.team_qb_baseline(qg, qbv, prior, season, week)
        for _, gm in gw.iterrows():
            h, a = TIDX[gm.home_team], TIDX[gm.away_team]
            row = {"game_id": gm.game_id}
            for k in ["off_epa", "def_epa", "off_sr", "def_sr", "off_pass", "def_pass", "off_rush", "def_rush"]:
                row[f"h_{k}"] = r[k][h]
                row[f"a_{k}"] = r[k][a]
            row["h_margin"] = r["margin"][h]
            row["a_margin"] = r["margin"][a]
            row["hfa_est"] = r["hfa_est"]
            for side, qid in [("h", gm.home_qb_id), ("a", gm.away_qb_id)]:
                team = gm.home_team if side == "h" else gm.away_team
                v = qbv.get(qid, prior) if isinstance(qid, str) else np.nan
                row[f"{side}_qb_val"] = v
                row[f"{side}_qb_base"] = base.get(team, prior)
            rows.append(row)
        if verbose and week == 1:
            print("built", season, flush=True)
    f = pd.DataFrame(rows)
    out = games.merge(f, on="game_id", how="inner")
    out = features.situational(out)
    # Derived differentials (home minus away), per-play -> per-game points scale (~62 plays)
    P = 62.0
    out["net_epa_diff"] = P * ((out.h_off_epa - out.h_def_epa) - (out.a_off_epa - out.a_def_epa))
    out["net_sr_diff"] = 100 * ((out.h_off_sr - out.h_def_sr) - (out.a_off_sr - out.a_def_sr))
    out["margin_diff"] = out.h_margin - out.a_margin
    # QB swap adjustment: (starter - baseline) EPA/dropback x ~36 dropbacks
    hq = (out.h_qb_val - out.h_qb_base).fillna(0)
    aq = (out.a_qb_val - out.a_qb_base).fillna(0)
    out["qb_adj_diff"] = 36.0 * (hq - aq)
    # Luck: points-based rating ahead of efficiency rating => expect regression
    out["luck_diff"] = out["margin_diff"] - out["net_epa_diff"]
    return out


if __name__ == "__main__":
    import sys
    last = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    df = build(last)
    df.to_parquet(f"{data.DATA_DIR}/features.parquet")
    print(df.shape)
