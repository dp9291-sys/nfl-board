"""Produce a week's projections + everything the dashboard needs, as JSON.

Usage:  python -m nflats.predict 2026 12      (after python -m nflats.build 2026)
"""
import json
import sys
import numpy as np
import pandas as pd
from . import data, ratings, model as M
from .ratings import TEAMS, TIDX

TEASER_BREAKEVEN = {"2-team -120": 0.7385, "2-team -110": 0.7237, "3-team +160": 0.7310}


def fit_final(df, season):
    tr = df[df.result.notna() & df.spread_line.notna() & (df.season < season)]
    mu, sd = tr[M.POWER_FEATS].mean(), tr[M.POWER_FEATS].std()
    b = M.ridge_fit(((tr[M.POWER_FEATS] - mu) / sd).values, tr.result.clip(-28, 28).values,
                    lam=20, w=M.season_w(tr.season, season))
    raw = b[1:] / sd.values
    icpt = b[0] - (raw * mu.values).sum()
    return icpt, dict(zip(M.POWER_FEATS, raw))


def backtest_summary(df, season):
    te = M.walk_forward(df, list(range(2018, season + 1)))
    k = float(te[te.season == season].k_slope.iloc[0])
    tab = M.ats_eval(te, "projA", (0, 1, 2, 3, 4)).to_dict("records")
    by = te.assign(e=te.projA - te.spread_line, a=te.result - te.spread_line).query("a != 0")
    by_season = by.groupby("season").apply(
        lambda x: {"n": int(len(x)), "hit": round(100 * float((np.sign(x.e) == np.sign(x.a)).mean()), 1)},
        include_groups=False).to_dict()
    mae = {"market": round(float((te.result - te.spread_line).abs().mean()), 2),
           "model": round(float((te.result - te.projA).abs().mean()), 2)}
    ks = te.groupby("season").k_slope.first().round(3).to_dict()
    return k, tab, {int(s): v for s, v in by_season.items()}, mae, {int(s): v for s, v in ks.items()}


def teaser_history(df, season):
    d = df[df.result.notna() & df.spread_line.notna() & (df.game_type == "REG") & (df.season >= 2018)
           & (df.season <= season)]
    out = {}
    for lab, cond, side in [("dog_home", d.spread_line.between(-2.5, -1.5), 1),
                            ("dog_away", d.spread_line.between(1.5, 2.5), -1),
                            ("fav_home", d.spread_line.between(7.5, 8.5), 1),
                            ("fav_away", d.spread_line.between(-8.5, -7.5), -1)]:
        x = d[cond]
        a = (x.result - x.spread_line) * side + 6
        out[lab] = {"n": int(len(x)), "win": round(float((a > 0).mean()), 3)}
    return out


def main(season, week):
    df = pd.read_parquet(f"{data.DATA_DIR}/features.parquet")
    games = data.load_games()
    pbp = data.load_pbp([season - 1, season])
    tg = data.team_game_stats(pbp)
    qg_all = data.qb_game_stats(data.load_pbp(list(range(season - 4, season + 1))))

    icpt, coef = fit_final(df, season)
    k, bt_tab, bt_season, mae, ks = backtest_summary(df, season)
    tabs = M.pmf_tables(df, season + 1)  # all completed games
    pmf = {f"{s:.1f}": [round(float(v), 6) for v in M.pmf_at(s, tabs)] for s in np.arange(-20, 20.5, 0.5)}

    r = ratings.team_ratings_at(tg, games, season, week)
    qbv, prior, names, league = ratings.qb_values_at(qg_all, season, week)
    base = ratings.team_qb_baseline(qg_all, qbv, prior, season, week)

    # team QB lists: anyone with dropbacks for the team this season or last
    recent = qg_all[(qg_all.season >= season - 1)]
    team_qbs = {}
    for t in TEAMS:
        q = recent[recent.posteam == t].groupby("passer_player_id").n.sum()
        q = q[q >= 15].index.tolist()
        team_qbs[t] = [{"id": i, "name": names.get(i, i), "val": round(float(qbv.get(i, prior)), 4),
                        "db": int(recent[(recent.posteam == t) & (recent.passer_player_id == i)].n.sum())}
                       for i in q]

    team_tbl = []
    for t in TEAMS:
        i = TIDX[t]
        team_tbl.append({"team": t,
                         "off_epa": round(float(r["off_epa"][i]), 4), "def_epa": round(float(r["def_epa"][i]), 4),
                         "off_sr": round(float(r["off_sr"][i]), 4), "def_sr": round(float(r["def_sr"][i]), 4),
                         "off_pass": round(float(r["off_pass"][i]), 4), "off_rush": round(float(r["off_rush"][i]), 4),
                         "def_pass": round(float(r["def_pass"][i]), 4), "def_rush": round(float(r["def_rush"][i]), 4),
                         "margin": round(float(r["margin"][i]), 2), "qb_base": round(float(base.get(t, prior)), 4)})

    wk = df[(df.season == season) & (df.week == week)]
    out_games = []
    sit_feats = ["hfa_flag", "rest_diff", "home_bye", "away_bye", "home_short", "away_short",
                 "tz_shift", "early_west", "div"]
    for g in wk.itertuples():
        contrib = {f: round(float(coef[f] * getattr(g, f)), 3) for f in M.POWER_FEATS}
        ratings_pts = contrib["net_epa_diff"] + contrib["net_sr_diff"] + contrib["margin_diff"]
        sit = {f: contrib[f] for f in sit_feats}
        hq = {"id": g.home_qb_id, "name": g.home_qb_name, "val": round(float(g.h_qb_val), 4)} if isinstance(g.home_qb_id, str) else None
        aq = {"id": g.away_qb_id, "name": g.away_qb_name, "val": round(float(g.a_qb_val), 4)} if isinstance(g.away_qb_id, str) else None
        out_games.append({
            "game_id": g.game_id, "gameday": str(g.gameday.date()), "weekday": g.weekday, "gametime": g.gametime,
            "home": g.home_team, "away": g.away_team, "stadium": g.stadium, "roof": g.roof if isinstance(g.roof, str) else None,
            "location": g.location,
            "spread_line": None if pd.isna(g.spread_line) else float(g.spread_line),
            "total_line": None if pd.isna(g.total_line) else float(g.total_line),
            "home_ml": None if pd.isna(g.home_moneyline) else int(g.home_moneyline),
            "away_ml": None if pd.isna(g.away_moneyline) else int(g.away_moneyline),
            "home_spread_odds": None if pd.isna(g.home_spread_odds) else int(g.home_spread_odds),
            "away_spread_odds": None if pd.isna(g.away_spread_odds) else int(g.away_spread_odds),
            "home_rest": int(g.home_rest), "away_rest": int(g.away_rest), "div": int(g.div),
            "ratings_pts": round(ratings_pts, 3), "intercept": round(icpt, 3), "situational": sit,
            "qb_base": {"home": round(float(g.h_qb_base), 4), "away": round(float(g.a_qb_base), 4)},
            "qb_start": {"home": hq, "away": aq},
            "luck_diff": round(float(g.luck_diff), 2),
        })

    payload = {
        "season": season, "week": week, "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "data_through": {"season": season, "week": int(tg[tg.season == season].week.max()),
                         "games": int(tg[tg.season == season].game_id.nunique())},
        "coef": {k_: round(float(v), 4) for k_, v in coef.items()}, "intercept": round(icpt, 3),
        "qb_dropbacks": 36.0, "qb_prior": round(float(prior), 4), "league_qb": round(float(league), 4),
        "blend_k": round(k, 3), "pmf": pmf, "teams": team_tbl, "team_qbs": team_qbs, "games": out_games,
        "backtest": {"table": bt_tab, "by_season": bt_season, "mae": mae, "k_by_season": ks,
                     "teasers": teaser_history(df, season), "teaser_breakeven": TEASER_BREAKEVEN},
    }
    path = f"{data.DATA_DIR}/week_{season}_{week:02d}.json"
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path, payload


if __name__ == "__main__":
    s, w = int(sys.argv[1]), int(sys.argv[2])
    path, p = main(s, w)
    print(path)
    for g in p["games"]:
        qb = p["coef"]["qb_adj_diff"] * 36 * (((g["qb_start"]["home"] or {}).get("val", p["qb_prior"]) - g["qb_base"]["home"])
                                              - ((g["qb_start"]["away"] or {}).get("val", p["qb_prior"]) - g["qb_base"]["away"]))
        proj = g["intercept"] + g["ratings_pts"] + sum(g["situational"].values()) + qb
        print(f'{g["away"]:>3} @ {g["home"]:<3} market {g["spread_line"]:+5.1f}  stats {proj:+5.1f}  qb {qb:+.1f}')
