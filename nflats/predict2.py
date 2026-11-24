"""v2 weekly export: python -m nflats.predict2 2026 12 [weather.json]

Needs: python -m nflats.build 2026 && python -m nflats.build2 2026 first.
Weather JSON (optional): {"games": {game_id: {"temp":..,"wind":..,"gust":..,"precip_pct":..,"rain_in":..,"snow_in":..,"phrase":..}}}
"""
import json
import sys
import warnings
import numpy as np
import pandas as pd
from . import data, ratings, advanced as A, model as M, model2 as M2
from .ratings import TEAMS, TIDX

warnings.filterwarnings("ignore")
INDOOR = {"dome", "closed"}
GROUPS = {
    "Team efficiency (EPA, success rate, margin, explosives, sacks)":
        ["net_epa_diff", "net_sr_diff", "margin_diff", "early_diff", "nto_diff", "pass_diff", "expl_diff", "sack_diff"],
    "Special teams": ["st_diff"],
    "Turnover rates, pass rate over expected, penalties": ["to_diff", "fum_diff", "proe_diff", "pen_diff"],
    "Listed QBs vs team baseline (EPA + CPOE)": ["qb_adj_diff", "qb_cpoe_diff"],
    "Injury report (snap-weighted)": M2.INJ,
    "Home field, rest, time zone, division": ["hfa_flag", "rest_diff", "home_bye", "away_bye", "home_short", "away_short",
                                              "tz_shift", "early_west", "div"],
    "Weather": ["cold_away_warm", "cold_home_warm", "wx_compress"],
}


def apply_weather(wk, wx):
    wk = wk.copy()
    for i, r in wk.iterrows():
        outdoor = (r.roof if isinstance(r.roof, str) else "closed") not in INDOOR
        w = wx.get(r.game_id) if outdoor else None
        wk.at[i, "wind_mph"] = float(w["wind"]) if w else 0.0
        wk.at[i, "temp_f"] = float(w["temp"]) if w else 68.0
    out = ~wk.roof.fillna("closed").isin(INDOOR)
    wk["cold_away_warm"] = (out & (wk.temp_f <= 40) & wk.away_team.isin(A.DOME_OR_WARM) & ~wk.home_team.isin(A.DOME_OR_WARM)).astype(float)
    wk["cold_home_warm"] = (out & (wk.temp_f <= 40) & wk.home_team.isin(A.DOME_OR_WARM) & ~wk.away_team.isin(A.DOME_OR_WARM)).astype(float)
    wk["wind15"] = (wk.wind_mph >= 15).astype(float)
    wk["wx_compress"] = wk.wind15 * wk.margin_diff
    return wk


def fit_power(df, season, feats):
    tr = df[(df.season >= 2013) & (df.season < season)]
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    b = M.ridge_fit(((tr[feats] - mu) / sd).values, tr.result.clip(-28, 28).values, lam=20, w=M.season_w(tr.season, season))
    raw = b[1:] / sd.values
    return b[0] - (raw * mu.values).sum(), dict(zip(feats, raw))


def week_injuries(season, week, coef):
    """Players on this week's report with snap share and estimated point value (if reports exist)."""
    inj, snaps, players = A.load_injury_inputs([season - 1, season])
    cur = inj[(inj.season == season) & (inj.week == week)]
    if cur.empty:
        return {}
    _, m = A.injury_features(cur, snaps, players)
    out = {}
    for r in m.itertuples():
        pts = -coef.get(f"inj_{r.grp}_diff", 0) * r.val  # value for the team missing him (positive = hurts team)
        out.setdefault(r.team, []).append({"name": r.full_name, "pos": r.position, "status": r.report_status,
                                           "snap": round(float(r.roll), 2), "pts": round(float(pts), 2)})
    for t in out:
        out[t].sort(key=lambda x: -x["pts"])
    return out


def main(season, week, wx_path=None):
    f1 = pd.read_parquet(f"{data.DATA_DIR}/features.parquet")
    f2 = pd.read_parquet(f"{data.DATA_DIR}/features2.parquet")
    df = M2.prep(f2)
    wx = json.load(open(wx_path))["games"] if wx_path else {}
    feats = list(M.POWER_FEATS) + sum([g for g in M2.GROUPS.values()], [])
    icpt, coef = fit_power(df, season, feats)

    # --- backtests (walk-forward, test 2018+) ---
    tab, projs, _, dfa = M2.ablation(f2)
    test = dfa.season >= 2018
    st_pred, st_coefs = M2.ats_model(dfa, dfa.spread_line.copy(), ["st_diff"], lam=50)
    st_tab = M2.ats_stats(dfa[test], st_pred[test], t=(0, 0.25, 0.5, 0.75, 1.0))
    c_st = float(st_coefs[season]["st_diff"])
    blocks = {}
    e, a = st_pred - dfa.spread_line, dfa.result - dfa.spread_line
    for per in [(2018, 2020), (2021, 2023), (2024, season)]:
        mm = test & (e.abs() > 0.5) & (a != 0) & dfa.season.between(*per)
        blocks[f"{per[0]}-{per[1]}"] = {"n": int(mm.sum()), "hit": round(100 * float((np.sign(e[mm]) == np.sign(a[mm])).mean()), 1)}
    gap_pred, gap_coefs = M2.ats_model(dfa, projs["+ weather"], [], lam=50)
    k_gap = max(0.0, float(gap_coefs[season]["gap"]))

    # --- key-number tables by scoring environment ---
    T = M.pmf_tables_by_total(dfa, season + 1)
    T["all"] = M.pmf_tables(dfa, season + 1)  # teaser legs: direct probabilities from all games
    pmf = {g: {f"{s:.1f}": [round(float(v), 6) for v in M.pmf_at(s, T[g])] for s in np.arange(-20, 20.5, 0.5)} for g in T}

    # --- this week ---
    wk = f2[(f2.season == season) & (f2.week == week)].copy()
    wk = apply_weather(wk, wx)
    games = data.load_games()
    pbp = A.load_pbp2([season - 1, season])
    tg2 = A.team_game_v2(pbp)
    r2 = A.ratings_v2_at(tg2, season, week)
    v1week = json.load(open(f"{data.DATA_DIR}/week_{season}_{week:02d}.json"))  # QB lists, team table, v1 pieces
    injuries = week_injuries(season, week, coef)

    out_games = []
    v1g = {g["game_id"]: g for g in v1week["games"]}
    for g in wk.itertuples():
        contrib = {k: float(coef[k] * getattr(g, k)) for k in feats}
        grp = {name: round(sum(contrib[k] for k in ks), 3) for name, ks in GROUPS.items()}
        base = v1g[g.game_id]
        w = wx.get(g.game_id) if (g.roof if isinstance(g.roof, str) else "closed") not in INDOOR else None
        et = M.effective_total(g.total_line, w["wind"] if w else 0)
        out_games.append({**{k: base[k] for k in ["game_id", "gameday", "weekday", "gametime", "home", "away", "stadium", "roof",
                                                  "location", "spread_line", "total_line", "home_ml", "away_ml", "home_spread_odds",
                                                  "away_spread_odds", "home_rest", "away_rest", "div", "qb_base", "qb_start", "luck_diff"]},
                          "intercept": round(icpt, 3), "groups": grp,
                          "inj_pts_model": grp["Injury report (snap-weighted)"],
                          "st_diff": round(float(g.st_diff), 3), "h_st": round(float(g.h2_off_st), 2), "a_st": round(float(g.a2_off_st), 2),
                          "weather": w, "eff_total": round(float(et), 1), "pmf_group": "lo" if et <= M.TOTAL_SPLIT else "hi",
                          "injuries": {"home": injuries.get(g.home_team, []), "away": injuries.get(g.away_team, [])},
                          "mkt_dev": round(float(g.mkt_dev), 2) if not pd.isna(g.mkt_dev) else None})

    teams = []
    for t in v1week["teams"]:
        i = TIDX[t["team"]]
        t = dict(t)
        for k in ["early", "expl", "sack", "nto"]:
            t[f"off_{k}"], t[f"def_{k}"] = round(float(r2[f"off_{k}"][i]), 4), round(float(r2[f"def_{k}"][i]), 4)
        t["st"] = round(float(r2["off_st"][i]), 2)
        t["int_rate"] = round(float(r2["off_int_rate"][i]), 4)
        t["takeaway_int"] = round(float(r2["def_int_rate"][i]), 4)
        t["proe"] = round(float(r2["off_proe"][i]), 3)
        t["cpoe"] = round(float(r2["off_cpoe"][i]), 2)
        teams.append(t)

    st_tab_v = {kk: vv for kk, vv in st_tab.items()}
    payload = {
        "version": 2, "season": season, "week": week, "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "data_through": v1week["data_through"], "coef_qb": round(float(coef["qb_adj_diff"]), 4),
        "qb_dropbacks": 36.0, "qb_prior": v1week["qb_prior"], "team_qbs": v1week["team_qbs"],
        "c_st": round(c_st, 3), "k_gap": round(k_gap, 3), "teams": teams, "games": out_games, "pmf": pmf,
        "weather_source": (json.load(open(wx_path)).get("source", "forecast") if wx_path else None),
        "inj_coef": {g: round(float(coef[f"inj_{g}_diff"]), 2) for g in A.INJ_GROUPS},
        "backtest": {
            "ablation": tab.fillna("").to_dict("records"),
            "st_layer": st_tab_v, "st_blocks": blocks,
            "st_coef_by_season": {int(s): round(float(v["st_diff"]), 3) for s, v in st_coefs.items()},
            "gap_coef_by_season": {int(s): round(float(v["gap"]), 3) for s, v in gap_coefs.items()},
            "teasers": v1week["backtest"]["teasers"], "by_season_v1": v1week["backtest"]["by_season"],
            "push3": {"low_total": 0.206, "high_total": 0.146},
        },
    }
    path = f"{data.DATA_DIR}/week2_{season}_{week:02d}.json"
    json.dump(payload, open(path, "w"), separators=(",", ":"))
    return path, payload


if __name__ == "__main__":
    s, w = int(sys.argv[1]), int(sys.argv[2])
    path, p = main(s, w, sys.argv[3] if len(sys.argv) > 3 else None)
    print(path, "c_st", p["c_st"], "k_gap", p["k_gap"])
    for g in p["games"]:
        stats = g["intercept"] + sum(g["groups"].values())
        proj = g["spread_line"] + p["c_st"] * g["st_diff"]
        print(f'{g["away"]:>3} @ {g["home"]:<3} mkt {g["spread_line"]:+5.1f} stats {stats:+5.1f} ST {g["st_diff"]:+.2f} proj {proj:+5.1f} '
              f'tot {g["eff_total"]} {g["pmf_group"]} wx {(g["weather"] or {}).get("wind","-")}')
