"""Inputs for the board's live injury fallback (python -m nflats.injlive PATH).

nflverse's injury file can lag the official reports by days on short weeks (Thanksgiving, Thursday games),
so the page also fetches a live source on the phone (ESPN, then Sleeper). To price those absences the same
way as the official report, the page needs each player's recent snap share; this adds:
  wk["snaps"]       = {team: [[name, position, snap share], ...]}  (last 6 games played, this season and last)
  wk["injury_week"] = latest week with an nflverse injury report for this season (None if none)
"""
import json
import os
import sys
import pandas as pd
from . import data


def snap_table(season, min_share=0.10):
    fr = []
    for y in (season - 1, season):
        p = os.path.join(data.DATA_DIR, f"snap_{y}.parquet")
        if os.path.exists(p):
            fr.append(pd.read_parquet(p, columns=["season", "week", "player", "pfr_player_id", "position", "team",
                                                  "offense_pct", "defense_pct"]))
    if not fr:
        return {}
    s = pd.concat(fr, ignore_index=True)
    s["team"] = s.team.replace(data.TEAM_MAP)
    s["pct"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s = s.sort_values(["season", "week"])
    last = s.groupby("pfr_player_id").tail(6)
    agg = last.groupby("pfr_player_id").agg(player=("player", "last"), position=("position", "last"),
                                            team=("team", "last"), share=("pct", "mean"), season=("season", "max"))
    agg = agg[(agg.share >= min_share)]
    out = {}
    for r in agg.sort_values("share", ascending=False).itertuples():
        out.setdefault(r.team, []).append([r.player, r.position, round(float(r.share), 2)])
    return out


def injury_week(season):
    from .health import inj_asof
    return inj_asof(season)


def enrich(path):
    wk = json.load(open(path))
    season = int(wk["season"])
    wk["snaps"] = snap_table(season)
    wk["injury_week"] = injury_week(season)
    json.dump(wk, open(path, "w"))
    return wk


if __name__ == "__main__":
    w = enrich(sys.argv[1])
    print(len(w["snaps"]), "teams;", sum(len(v) for v in w["snaps"].values()), "players; injury report week", w["injury_week"])
