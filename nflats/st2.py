"""Special teams v2 candidates (python -m nflats.st2).

All ratings use only games before the week being predicted.
  fg / punt / ko   : net EPA per game for each phase (decayed, shrunk). Kickoff rating resets at the
                     2024 dynamic-kickoff rule change, so post-rule ratings use only post-rule games.
  ko_post24        : kickoff differential x (season >= 2024), letting the kickoff weight change with the rule
  kick_val         : each team's game-day kicker, rated from his own FG/XP history (logit skill, shrunk),
                     priced in that game's wind, cold and altitude over a typical attempt mix -> points/game
  punt_val         : game-day punter's own net EPA per punt (shrunk) x punts per game
  ret_out          : primary kick/punt returner (most returns over prior 3 games) listed Out/Doubtful
                     -> loses his return value vs league average
Outputs data/st2.parquet with home-minus-away differentials per game.
"""
import os
import sys
import numpy as np
import pandas as pd
from . import data
from .ratings import TEAMS, TIDX, _weights

FIRST = 2009
COLS = ["game_id", "season", "week", "posteam", "defteam", "home_team", "away_team", "play_type", "special_teams_play",
        "epa", "kick_distance", "field_goal_result", "extra_point_result", "kicker_player_id", "punter_player_id",
        "kickoff_returner_player_id", "punt_returner_player_id", "temp", "wind", "roof", "stadium"]
HIGH_ALT = {"DEN"}


def load(last):
    pbp = pd.concat([pd.read_parquet(os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"), columns=COLS)
                     for y in range(FIRST, last + 1)], ignore_index=True)
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        pbp[c] = pbp[c].replace(data.TEAM_MAP)
    return pbp


def game_weather(pbp):
    g = pbp.groupby("game_id").agg(temp=("temp", "first"), wind=("wind", "first"), roof=("roof", "first"),
                                   home_team=("home_team", "first"), season=("season", "first"), week=("week", "first"))
    indoor = g.roof.isin(["dome", "closed"])
    g["wind_eff"] = np.where(indoor, 0.0, g.wind.fillna(6.0))
    g["cold"] = np.where(indoor, 0.0, np.clip(50 - g.temp.fillna(60), 0, None) / 10)  # per 10 F below 50
    g["alt"] = g.home_team.isin(HIGH_ALT).astype(float)
    return g


def phase_net(pbp):
    st = pbp[(pbp.special_teams_play == 1) & pbp.epa.notna() & pbp.posteam.notna()]
    st = st.assign(kind=np.select([st.play_type.isin(["field_goal", "extra_point"]), st.play_type == "punt",
                                   st.play_type == "kickoff"], ["fg", "punt", "ko"], "other"))
    a = st.groupby(["game_id", "posteam", "kind"]).epa.sum()
    b = st.groupby(["game_id", "defteam", "kind"]).epa.sum().rename_axis(["game_id", "posteam", "kind"])
    net = a.sub(b, fill_value=0).unstack("kind", fill_value=0).reset_index()
    for k in ["fg", "punt", "ko"]:
        if k not in net:
            net[k] = 0.0
    keys = pbp.groupby(["game_id", "posteam"]).agg(season=("season", "first"), week=("week", "first")).reset_index()
    return keys.merge(net[["game_id", "posteam", "fg", "punt", "ko"]], on=["game_id", "posteam"], how="left").fillna(0)


def decayed(tg, col, season, week, K=4.0, min_season=None):
    prev_max = tg.loc[tg.season == season - 1, "week"].max()
    m = ((tg.season == season) & (tg.week < week)) | (tg.season == season - 1)
    if min_season is not None:
        m &= tg.season >= min_season
    d = tg[m]
    if not len(d):
        return np.zeros(len(TEAMS))
    w = _weights(d, season, week, prev_max if prev_max == prev_max else 18)
    s = pd.Series(w * d[col].values).groupby(d.posteam.values).sum()
    ws = pd.Series(w).groupby(d.posteam.values).sum()
    lg = float((w * d[col].values).sum() / w.sum())
    return ((s + K * lg) / (ws + K)).reindex(TEAMS).fillna(lg).values


def _logit_fit(X, y, w, lam=1e-3, iters=30):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = w * p * (1 - p)
        H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1])
        g = X.T @ (w * (y - p)) - lam * b
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def _fgX(dist, wind, cold, alt):
    d = (dist - 40) / 10
    return np.c_[np.ones_like(d), d, d * d, wind / 10, (wind / 10) * np.clip(d, 0, None), cold, alt]


def kicker_tables(pbp, wx):
    fg = pbp[(pbp.play_type == "field_goal") & pbp.kick_distance.notna() & pbp.kicker_player_id.notna()].copy()
    fg = fg.join(wx[["wind_eff", "cold", "alt"]], on="game_id")
    fg["make"] = (fg.field_goal_result == "made").astype(float)
    xp = pbp[(pbp.play_type == "extra_point") & pbp.kicker_player_id.notna()].copy()
    xp["make"] = (xp.extra_point_result == "good").astype(float)
    return fg, xp


def kicker_model(fg, xp, season, week, lam_k=25.0):
    """League FG curve (prior 8 seasons) + kicker logit skill from his own decayed attempts (FG + XP)."""
    hist = fg[(fg.season < season) | ((fg.season == season) & (fg.week < week))]
    lg = hist[hist.season >= season - 8]
    X = _fgX(lg.kick_distance.values.astype(float), lg.wind_eff.values, lg.cold.values, lg.alt.values)
    b = _logit_fit(X, lg.make.values, np.ones(len(lg)))
    # skill: one-step Newton logit offset, decayed by age in seasons
    h = hist[hist.season >= season - 5]
    Xh = _fgX(h.kick_distance.values.astype(float), h.wind_eff.values, h.cold.values, h.alt.values)
    p = 1 / (1 + np.exp(-(Xh @ b)))
    age = (season - h.season.values) + np.where(h.season.values == season, 0, 0.5)
    w = 0.6 ** age
    num = pd.Series(w * (h.make.values - p)).groupby(h.kicker_player_id.values).sum()
    den = pd.Series(w * p * (1 - p)).groupby(h.kicker_player_id.values).sum()
    xh = xp[((xp.season < season) | ((xp.season == season) & (xp.week < week))) & (xp.season >= season - 5)]
    if len(xh):
        pxp = xh.make.mean()
        ax = (season - xh.season.values) + np.where(xh.season.values == season, 0, 0.5)
        wx_ = 0.6 ** ax
        num = num.add(pd.Series(wx_ * (xh.make.values - pxp)).groupby(xh.kicker_player_id.values).sum(), fill_value=0)
        den = den.add(pd.Series(wx_ * pxp * (1 - pxp)).groupby(xh.kicker_player_id.values).sum(), fill_value=0)
    skill = (num / (den + lam_k)).to_dict()
    mix = lg.kick_distance.value_counts(normalize=True).sort_index()
    return b, skill, mix


def kick_value(b, skill_k, mix, wind, cold, alt, fga_pg=1.8, xpa_pg=2.3):
    d = mix.index.values.astype(float)
    X = _fgX(d, np.full_like(d, wind), np.full_like(d, cold), np.full_like(d, alt))
    base = X @ b
    p1 = 1 / (1 + np.exp(-(base + skill_k)))
    p0 = 1 / (1 + np.exp(-base))
    fg = fga_pg * 3 * float(np.sum(mix.values * (p1 - p0)))
    xb = np.log(0.94 / 0.06)
    xpv = xpa_pg * float(1 / (1 + np.exp(-(xb + skill_k))) - 0.94)
    return fg + xpv


def player_rates(df, idcol, valcol, season, week, K):
    h = df[((df.season < season) | ((df.season == season) & (df.week < week))) & (df.season >= season - 3) & df[idcol].notna()]
    if not len(h):
        return {}
    age = (season - h.season.values) + np.where(h.season.values == season, 0, 0.5)
    w = 0.6 ** age
    lg = float(np.sum(w * h[valcol].values) / np.sum(w))
    s = pd.Series(w * (h[valcol].values - lg)).groupby(h[idcol].values).sum()
    n = pd.Series(w).groupby(h[idcol].values).sum()
    return (s / (n + K)).to_dict()


def injuries_out(last):
    frames = []
    for y in range(2012, last + 1):
        p = os.path.join(data.DATA_DIR, f"inj_{y}.parquet")
        if os.path.exists(p):
            x = pd.read_parquet(p, columns=["season", "week", "team", "gsis_id", "report_status"])
            frames.append(x[x.report_status.isin(["Out", "Doubtful"])])
    if not frames:
        return set()
    x = pd.concat(frames)
    x["team"] = x.team.replace(data.TEAM_MAP)
    return set(zip(x.season, x.week, x.gsis_id))


def build(last):
    pbp = load(last)
    wx = game_weather(pbp)
    tg = phase_net(pbp)
    fg, xp = kicker_tables(pbp, wx)
    punts = pbp[(pbp.play_type == "punt") & pbp.epa.notna() & pbp.punter_player_id.notna()]
    kr = pbp[(pbp.play_type == "kickoff") & pbp.epa.notna() & pbp.kickoff_returner_player_id.notna()].assign(val=lambda x: x.epa)
    pr = pbp[(pbp.play_type == "punt") & pbp.epa.notna() & pbp.punt_returner_player_id.notna()].assign(val=lambda x: -x.epa)
    outs = injuries_out(last)
    games = data.load_games()
    games = games[(games.season >= 2012) & (games.season <= last) & games.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])]
    # game-day kicker / punter per team-game (from the game itself, else most recent)
    gk = fg.groupby(["game_id", "posteam"]).kicker_player_id.agg(lambda s: s.mode().iloc[0])
    gk = gk.combine_first(xp.groupby(["game_id", "posteam"]).kicker_player_id.agg(lambda s: s.mode().iloc[0]))
    gp = punts.groupby(["game_id", "posteam"]).punter_player_id.agg(lambda s: s.mode().iloc[0])
    kr_g = kr.groupby(["game_id", "posteam"]).kickoff_returner_player_id.agg(list)   # posteam = receiving team on kickoffs
    pr_g = pr.groupby(["game_id", "defteam"]).punt_returner_player_id.agg(list)
    order = games.sort_values(["season", "week"])[["game_id", "season", "week", "home_team", "away_team"]]
    team_hist = {t: [] for t in TEAMS}
    rows = []
    for (season, week), wk in order.groupby(["season", "week"], sort=True):
        if season < 2013:
            for r in wk.itertuples():
                for t in (r.home_team, r.away_team):
                    team_hist[t].append(r.game_id)
            continue
        R = {c: decayed(tg, c, season, week) for c in ["fg", "punt", "ko"]}
        R["ko_reset"] = decayed(tg, "ko", season, week, min_season=2024) if season >= 2024 else R["ko"]
        b, skill, mix = kicker_model(fg, xp, season, week)
        prate = player_rates(punts, "punter_player_id", "epa", season, week, K=40)
        krate = player_rates(kr, "kickoff_returner_player_id", "val", season, week, K=25)
        prrate = player_rates(pr, "punt_returner_player_id", "val", season, week, K=25)
        for r in wk.itertuples():
            w = wx.loc[r.game_id] if r.game_id in wx.index else None
            wind = float(w.wind_eff) if w is not None else 6.0
            cold = float(w.cold) if w is not None else 0.0
            alt = float(w.alt) if w is not None else float(r.home_team in HIGH_ALT)
            vals = {}
            for side, t in (("h", r.home_team), ("a", r.away_team)):
                prev = team_hist[t][-3:]
                k = gk.get((r.game_id, t))
                if k is None:  # fall back to the team's most recent kicker
                    for g in reversed(team_hist[t]):
                        if (g, t) in gk.index:
                            k = gk[(g, t)]
                            break
                p = gp.get((r.game_id, t))
                if p is None:
                    for g in reversed(team_hist[t]):
                        if (g, t) in gp.index:
                            p = gp[(g, t)]
                            break
                vals[f"{side}_kick"] = kick_value(b, skill.get(k, 0.0), mix, wind, cold, alt)
                vals[f"{side}_punt"] = 4.3 * prate.get(p, 0.0)
                # primary returners over the prior 3 games
                loss = 0.0
                for gl, rate, per_g in ((kr_g, krate, 2.5), (pr_g, prrate, 2.0)):
                    ids = [x for g in prev for x in gl.get((g, t), [])]
                    if ids:
                        top = pd.Series(ids).value_counts().index[0]
                        if (season, week, top) in outs:
                            loss -= per_g * rate.get(top, 0.0)
                vals[f"{side}_ret_out"] = loss
            h, a = TIDX[r.home_team], TIDX[r.away_team]
            post = float(season >= 2024)
            rows.append({"game_id": r.game_id, "season": season, "week": week,
                         "st2_fg": R["fg"][h] - R["fg"][a], "st2_punt": R["punt"][h] - R["punt"][a],
                         "st2_ko": R["ko_reset"][h] - R["ko_reset"][a],
                         "st2_ko_post24": (R["ko_reset"][h] - R["ko_reset"][a]) * post,
                         "kick_val": vals["h_kick"] - vals["a_kick"], "punt_val": vals["h_punt"] - vals["a_punt"],
                         "ret_out": vals["h_ret_out"] - vals["a_ret_out"]})
            for t in (r.home_team, r.away_team):
                team_hist[t].append(r.game_id)
        if week == 1:
            print(f"  st2 {season}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(data.DATA_DIR, "st2.parquet"))
    return out


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
