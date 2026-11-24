"""Game-day inactives (posted ~90 min before kickoff) from nflverse weekly rosters (status INA).

    python -m nflats.inactives check     -> prints "BUILD <teams>" or "SKIP <reason>"
Merging: players marked INA for the current week are added to that week's injury report as Out
(or upgraded to Out if they were listed Questionable/Doubtful), so every downstream injury and
player-availability calculation treats them as not playing. INA flags only appear on game day.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from . import data

REL = "https://github.com/nflverse/nflverse-data/releases/download"
ET = ZoneInfo("America/New_York")
STATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inactives_state.json")


def roster_path(season):
    return os.path.join(data.DATA_DIR, f"roster_weekly_{season}.parquet")


def download_roster(season):
    os.makedirs(data.DATA_DIR, exist_ok=True)
    tmp = roster_path(season) + ".part"
    subprocess.run(["curl", "-sfL", "-o", tmp, f"{REL}/weekly_rosters/roster_weekly_{season}.parquet"], check=True)
    os.replace(tmp, roster_path(season))


def inactive_players(season, week, teams=None):
    p = roster_path(season)
    if not os.path.exists(p):
        return pd.DataFrame(columns=["team", "gsis_id", "full_name", "position"])
    r = pd.read_parquet(p)
    r["team"] = r.team.replace(data.TEAM_MAP)
    r = r[(r.week == week) & (r.status == "INA") & r.gsis_id.notna()]
    if teams:
        r = r[r.team.isin(teams)]
    return r[["team", "gsis_id", "full_name", "first_name", "last_name", "position"]]


def merge_into_injuries(season, week, log=print):
    ina = inactive_players(season, week)
    path = os.path.join(data.DATA_DIR, f"inj_{season}.parquet")
    if ina.empty or not os.path.exists(path):
        return 0
    inj = pd.read_parquet(path)
    wk = (inj.season == season) & (inj.week == week)
    upgrade = wk & inj.gsis_id.isin(ina.gsis_id) & (inj.report_status != "Out")
    inj.loc[upgrade, "report_status"] = "Out"
    have = set(inj.loc[wk, "gsis_id"])
    add = ina[~ina.gsis_id.isin(have)]
    rows = pd.DataFrame({c: [None] * len(add) for c in inj.columns})
    for c, v in {"season": season, "season_type": "REG", "game_type": "REG", "week": week,
                 "report_status": "Out", "report_primary_injury": "Inactive (game day)"}.items():
        if c in rows:
            rows[c] = v
    for c in ["team", "gsis_id", "full_name", "first_name", "last_name", "position"]:
        if c in rows:
            rows[c] = add[c].values
    out = pd.concat([inj, rows.astype(inj.dtypes.to_dict(), errors="ignore")], ignore_index=True)
    out.to_parquet(path)
    log(f"  inactives: {len(add)} added, {int(upgrade.sum())} upgraded to Out")
    return len(add) + int(upgrade.sum())


def check(now=None):
    """Decide whether a game-day rebuild is worthwhile right now."""
    now = now or datetime.now(timezone.utc)
    os.makedirs(data.DATA_DIR, exist_ok=True)
    subprocess.run(["curl", "-sfL", "-o", os.path.join(data.DATA_DIR, "games.csv"), data.GAMES_URL], check=True)
    g = data.load_games()
    g = g[g.result.isna() & g.gametime.notna()]
    soon = []
    for r in g.itertuples():
        hh, mm = map(int, str(r.gametime).split(":"))
        ko = datetime(r.gameday.year, r.gameday.month, r.gameday.day, hh, mm, tzinfo=ET)
        if now + timedelta(minutes=20) <= ko <= now + timedelta(minutes=100):
            soon.append(r)
    if not soon:
        return "SKIP no kickoff in the next 20-100 minutes"
    season, week = int(soon[0].season), int(soon[0].week)
    teams = sorted({t for r in soon for t in (r.home_team, r.away_team)})
    download_roster(season)
    ina = inactive_players(season, week, teams)
    if ina.empty:
        return f"SKIP inactives not posted yet for {' '.join(teams)}"
    sig = hashlib.sha1(",".join(sorted(ina.gsis_id)).encode()).hexdigest()
    prev = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if prev.get("sig") == sig:
        return "SKIP already rebuilt with these inactives"
    json.dump({"sig": sig, "season": season, "week": week, "teams": teams, "n": int(len(ina)),
               "at": now.isoformat(timespec="minutes")}, open(STATE, "w"))
    return f"BUILD {' '.join(teams)} ({len(ina)} inactives posted)"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print(check())
