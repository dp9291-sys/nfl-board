"""Special-teams breakdown for the week file (explanatory only; the model uses the total).

Per team, using the same decayed, shrunk ratings as the model (prior season + this season before this week):
  fg   : field goals and extra points vs expected (points per game)
  punt : punting, coverage and punt returns, net (field position -> expected points)
  ko   : kickoffs, coverage and kick returns, net
  other: everything else so the parts add to the model's number (fakes, penalties, safeties ...)
Flags: early season (rating leans on last season), new kicker / punter since most of the rating was built,
one big play (return TD or block) supplying a large share, edge mostly from kicking (the least repeatable part).
"""
import json
import os
import numpy as np
import pandas as pd
from . import data, st2
from .ratings import TEAMS, TIDX, _weights

COLS = st2.COLS + ["touchdown", "punt_blocked"]


def _load(S):
    fr = []
    for y in (S - 1, S):
        p = os.path.join(data.DATA_DIR, f"pbp_{y}.parquet")
        if os.path.exists(p):
            fr.append(pd.read_parquet(p, columns=COLS))
    pbp = pd.concat(fr, ignore_index=True)
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        pbp[c] = pbp[c].replace(data.TEAM_MAP)
    return pbp


def _label(r):
    if r.punt_blocked == 1:
        return "blocked punt"
    if r.field_goal_result == "blocked":
        return "blocked field goal"
    if r.touchdown == 1 and r.play_type == "kickoff":
        return "kickoff return TD"
    if r.touchdown == 1 and r.play_type == "punt":
        return "punt return TD"
    if r.play_type == "field_goal":
        return "missed field goal" if r.field_goal_result == "missed" else "field goal"
    if r.play_type == "kickoff":
        return "kickoff play (long return or lost fumble)"
    if r.play_type == "punt":
        return "punt play (long return, muff or fumble)"
    return r.play_type.replace("_", " ")


def breakdown(S, W):
    pbp = _load(S)
    tg = st2.phase_net(pbp)
    st = pbp[(pbp.special_teams_play == 1) & pbp.epa.notna() & pbp.posteam.notna()]
    a = st.groupby(["game_id", "posteam"]).epa.sum()
    b = st.groupby(["game_id", "defteam"]).epa.sum().rename_axis(["game_id", "posteam"])
    tg = tg.merge(a.sub(b, fill_value=0).rename("total").reset_index(), on=["game_id", "posteam"], how="left").fillna({"total": 0})
    tg["other"] = tg.total - tg.fg - tg.punt - tg.ko
    R = {c: st2.decayed(tg, c, S, W) for c in ["fg", "punt", "ko", "other", "total"]}
    # window weights for the big-play share
    prev_max = tg.loc[tg.season == S - 1, "week"].max()
    win = tg[((tg.season == S) & (tg.week < W)) | (tg.season == S - 1)]
    wts = pd.Series(_weights(win, S, W, prev_max if prev_max == prev_max else 18), index=win.game_id + "|" + win.posteam)
    stw = st[((st.season == S) & (st.week < W)) | (st.season == S - 1)].copy()
    fg = pbp[pbp.play_type.isin(["field_goal", "extra_point"]) & pbp.kicker_player_id.notna() & (((pbp.season == S) & (pbp.week < W)) | (pbp.season == S - 1))]
    pu = pbp[(pbp.play_type == "punt") & pbp.punter_player_id.notna() & (((pbp.season == S) & (pbp.week < W)) | (pbp.season == S - 1))]
    out = {}
    for t in TEAMS:
        i = TIDX[t]
        games_now = int(((tg.season == S) & (tg.week < W) & (tg.posteam == t)).sum())
        flags = []
        if games_now < 4:
            flags.append(f"early season ({games_now} games this year; the rating leans on last season)")
        for df, col, lab in ((fg, "kicker_player_id", "kicker"), (pu, "punter_player_id", "punter")):
            x = df[df.posteam == t]
            if len(x) < 5:
                continue
            gids = x.drop_duplicates("game_id").sort_values(["season", "week"]).game_id.tolist()[-2:]
            recent = x[x.game_id.isin(gids)][col].value_counts().index[0]      # who handled the last two games
            w = np.where(x.season == S, 1.0, 0.55)                               # this season counts more, like the rating
            share = float(w[(x[col] == recent).values].sum() / w.sum())
            if share < 0.4:
                flags.append(f"new {lab} ({round(100 * share)}% of the attempts behind the rating)")
        # biggest single play for this team (gain as kicking/receiving team, or loss as the other side)
        s_own = stw[stw.posteam == t].assign(v=lambda z: z.epa)
        s_opp = stw[stw.defteam == t].assign(v=lambda z: -z.epa)
        allp = pd.concat([s_own, s_opp])
        if len(allp):
            allp["w"] = (allp.game_id + "|" + t).map(wts).fillna(0)
            wsum = float(wts[wts.index.str.endswith("|" + t)].sum())
            if wsum > 0:
                k = (allp.w * allp.v).abs().idxmax()
                top = allp.loc[k]
                per_game = float(top.w * top.v / (wsum + 4.0))
                if abs(per_game) >= 0.25 and abs(R["total"][i]) >= 0.3 and np.sign(per_game) == np.sign(R["total"][i]):
                    when = f"week {int(top.week)}" if int(top.season) == S else "last season"
                    flags.append(f"{when}: one {_label(top)} supplies {abs(per_game):.1f} pts of the rating")
        tot = R["total"][i]
        if abs(tot) >= 0.5 and abs(R["fg"][i]) > 0.5 * abs(tot) and np.sign(R["fg"][i]) == np.sign(tot):
            flags.append("rating is mostly kicking accuracy, the least repeatable part")
        out[t] = {"fg": round(float(R["fg"][i]), 2), "punt": round(float(R["punt"][i]), 2), "ko": round(float(R["ko"][i]), 2),
                  "other": round(float(R["other"][i]), 2), "total": round(float(tot), 2), "games": games_now, "flags": flags}
    return out


def enrich(path):
    wk = json.load(open(path))
    wk["st_breakdown"] = breakdown(int(wk["season"]), int(wk["week"]))
    json.dump(wk, open(path, "w"), separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return wk["st_breakdown"]
