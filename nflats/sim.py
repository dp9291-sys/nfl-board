"""Drive-level game simulator (python -m nflats.sim).

For each week, using only earlier games (decayed: 9-week half-life, prior season x0.55; shrunk):
  offense / defense drive-outcome rates: touchdown, field goal made, turnover (incl. downs), opponent score
  pace: drives per game for each team
Each matchup combines offense and defense rates by odds ratios against the league rate, then simulates
20,000 games: drives ~ Poisson around the two teams' pace (same count for both, +-1 for who gets the ball
first), each drive drawn from the matchup's outcome probabilities; touchdowns score 6.96 (PAT/2-pt mix),
field goals 3, defensive scores 6.96 to the defense. Home field: league home margin over the prior 3 seasons.
Outputs data/sim.parquet: simulated mean margin and P(home covers / push / away covers) at the closing spread.
"""
import os
import sys
import numpy as np
import pandas as pd
from . import data
from .ratings import TEAMS, _weights

OUTC = ["td", "fg", "to", "opp"]  # remaining probability = no score
K_DRV = 40.0
TD_PTS = 6.96
N_SIM = 20000


def drives(first=2015, last=2026):
    fr = []
    for y in range(first, last + 1):
        p = pd.read_parquet(os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"),
                            columns=["game_id", "season", "week", "posteam", "defteam", "fixed_drive", "fixed_drive_result"])
        p = p.dropna(subset=["posteam", "fixed_drive_result"])
        dr = p.groupby(["game_id", "fixed_drive"]).agg(season=("season", "first"), week=("week", "first"),
                                                       posteam=("posteam", "first"), defteam=("defteam", "first"),
                                                       res=("fixed_drive_result", "first")).reset_index()
        fr.append(dr)
    d = pd.concat(fr, ignore_index=True)
    for c in ["posteam", "defteam"]:
        d[c] = d[c].replace(data.TEAM_MAP)
    d = d[d.res != "End of half"]
    d["td"] = (d.res == "Touchdown").astype(float)
    d["fg"] = (d.res == "Field goal").astype(float)
    d["to"] = d.res.isin(["Turnover", "Turnover on downs"]).astype(float)
    d["opp"] = d.res.isin(["Opp touchdown", "Safety"]).astype(float)
    return d


def week_rates(D, season, week, prev_max):
    m = ((D.season == season) & (D.week < week)) | (D.season == season - 1)
    d = D[m]
    w = _weights(d, season, week, prev_max)
    out = {"L": {k: float(np.average(d[k], weights=w)) for k in OUTC}}
    for side, key in (("o", "posteam"), ("d", "defteam")):
        n = pd.Series(w).groupby(d[key].values).sum()
        for k in OUTC:
            s = pd.Series(w * d[k].values).groupby(d[key].values).sum()
            out[f"{side}_{k}"] = np.array(((s + K_DRV * out["L"][k]) / (n + K_DRV)).reindex(TEAMS).fillna(out["L"][k]).values, float)
    # pace: drives per team-game (both teams' drives in their games)
    g = d.groupby(["game_id", "posteam"]).size().rename("n").reset_index()
    gw = d.groupby(["game_id", "posteam"]).apply(lambda x: 1.0, include_groups=False)  # placeholder for alignment
    gm = d.groupby("game_id").size() / 2.0
    games = d.drop_duplicates("game_id")[["game_id", "season", "week"]].set_index("game_id")
    wg = _weights(games.reset_index(), season, week, prev_max)
    pace_game = pd.Series(gm.reindex(games.index).values, index=games.index)
    teams = pd.concat([d[["game_id", "posteam"]].rename(columns={"posteam": "t"}), d[["game_id", "defteam"]].rename(columns={"defteam": "t"})]).drop_duplicates()
    teams = teams.merge(pd.DataFrame({"game_id": games.index, "pw": wg, "pace": pace_game.values}), on="game_id")
    lg = float(np.average(pace_game.values, weights=wg))
    s = (teams.pw * teams.pace).groupby(teams.t).sum()
    n = teams.pw.groupby(teams.t).sum()
    out["pace"] = np.array(((s + 6 * lg) / (n + 6)).reindex(TEAMS).fillna(lg).values, float)
    out["L"]["pace"] = lg
    return out


def _combine(L, o, dd):
    lo = np.log(L / (1 - L))
    return 1 / (1 + np.exp(-(np.log(o / (1 - o)) + np.log(dd / (1 - dd)) - lo)))


def probs(R, oi, di):
    p = np.array([_combine(R["L"][k], R[f"o_{k}"][oi], R[f"d_{k}"][di]) for k in OUTC])
    tot = p.sum()
    if tot > 0.95:
        p *= 0.95 / tot
    return np.r_[p, 1 - p.sum()]


def simulate(ph, pa, n_drives, hfa, line, rng):
    n = rng.poisson(n_drives, N_SIM)
    extra = rng.integers(0, 2, N_SIM)
    nh, na = n + extra, n + (1 - extra)
    pts = np.array([TD_PTS, 3.0, 0.0, -TD_PTS, 0.0])  # opp score counts against the offense
    def team_pts(p, k):
        mx = int(k.max())
        draws = rng.choice(5, size=(N_SIM, mx), p=p)
        mask = np.arange(mx)[None, :] < k[:, None]
        own = np.where(mask, np.where(draws <= 1, pts[draws], 0.0), 0.0).sum(1)
        opp = np.where(mask & (draws == 3), TD_PTS, 0.0).sum(1)
        return own, opp
    hp, h_opp = team_pts(ph, nh)
    ap, a_opp = team_pts(pa, na)
    margin = (hp + a_opp) - (ap + h_opp) + hfa
    m = np.round(margin)
    return float(margin.mean()), float((m > line).mean()), float((m == line).mean()), float((m < line).mean())


def build(last=2026, seed=7):
    D = drives(2015, last)
    g = data.load_games()
    g = g[(g.season >= 2016) & (g.season <= last)]
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(TEAMS)}
    rows = []
    for (S, W), wk in g.sort_values(["season", "week"]).groupby(["season", "week"]):
        prev_max = D.loc[D.season == S - 1, "week"].max()
        R = week_rates(D, S, W, prev_max if prev_max == prev_max else 18)
        past = g[(g.season.between(S - 3, S - 1)) & g.result.notna() & (g.game_type == "REG")]
        hfa = float(past.result.mean()) if len(past) else 1.5
        for r in wk.itertuples():
            if r.home_team not in idx or r.away_team not in idx:
                continue
            h, a = idx[r.home_team], idx[r.away_team]
            ph, pa = probs(R, h, a), probs(R, a, h)
            nd = (R["pace"][h] + R["pace"][a]) / 2 - 0.5
            line = r.spread_line if r.spread_line == r.spread_line else 0.0
            mean, ch, cp, ca = simulate(ph, pa, nd, hfa if r.location == "Home" else 0.0, line, rng)
            rows.append({"game_id": r.game_id, "season": S, "week": W, "sim_margin": mean, "sim_home_cover": ch,
                         "sim_push": cp, "sim_away_cover": ca, "sim_drives": nd,
                         "h_td": ph[0], "a_td": pa[0], "h_fg": ph[1], "a_fg": pa[1], "h_to": ph[2], "a_to": pa[2]})
        if W == 1:
            print("  sim", S, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(data.DATA_DIR, "sim.parquet"))
    return out


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
