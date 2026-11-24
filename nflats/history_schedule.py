"""Build the snapshot plan for the line-history download (python -m nflats.history_schedule OUT.json).

Per NFL week:
  open  : Tuesday 17:00 UTC before the week's first kickoff (first stable look-ahead number)
  fri   : Friday 17:00 UTC before the weekend games (midweek, after most injury news)
  close : 10 minutes before each distinct kickoff time
Each snapshot is one call to /v4/historical/sports/americanfootball_nfl/odds returning every listed game.
"""
import json
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from . import data

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "bovada", "betonlineag",
         "lowvig", "espnbet"]  # 10 books = 1 request-equivalent per market (20 credits per historical call)


def build(first=2020, last=None):
    g = data.load_games()
    g = g[(g.season >= first) & g.gametime.notna() & g.result.notna()]
    if last:
        g = g[g.season <= last]
    games, snaps = {}, []
    for r in g.itertuples():
        hh, mm = map(int, str(r.gametime).split(":"))
        ko = datetime(r.gameday.year, r.gameday.month, r.gameday.day, hh, mm, tzinfo=ET).astimezone(UTC)
        if ko < datetime(2020, 6, 10, tzinfo=UTC):  # history starts June 2020
            continue
        games[r.game_id] = [int(r.season), int(r.week), r.home_team, r.away_team, ko.strftime("%Y-%m-%dT%H:%M:%SZ")]
    by_week = {}
    for gid, (s, w, h, a, ko) in games.items():
        by_week.setdefault((s, w), []).append((gid, datetime.strptime(ko, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)))
    for (s, w), lst in sorted(by_week.items()):
        first_ko = min(k for _, k in lst)
        d = first_ko.astimezone(ET).date()
        tue = d - timedelta(days=(d.weekday() - 1) % 7 or 7)
        t_open = datetime(tue.year, tue.month, tue.day, 17, tzinfo=UTC)
        if t_open > first_ko - timedelta(hours=24):
            t_open = first_ko - timedelta(hours=48)
        snaps.append([f"{s}-{w:02d}-open", t_open.strftime("%Y-%m-%dT%H:%M:%SZ"), "open", s, [gid for gid, _ in lst]])
        sun = max(k for _, k in lst).astimezone(ET).date()
        sun = sun - timedelta(days=(sun.weekday() - 6) % 7)  # the week's Sunday
        fri = sun - timedelta(days=2)
        t_fri = datetime(fri.year, fri.month, fri.day, 17, tzinfo=UTC)
        later = [gid for gid, k in lst if k > t_fri + timedelta(hours=2)]
        if later and t_fri > t_open + timedelta(hours=24):
            snaps.append([f"{s}-{w:02d}-fri", t_fri.strftime("%Y-%m-%dT%H:%M:%SZ"), "fri", s, later])
        for k in sorted({k for _, k in lst}):
            snaps.append([f"{s}-{w:02d}-close-{k.strftime('%m%d%H%M')}", (k - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "close", s, [gid for gid, kk in lst if kk == k]])
    return {"v": 1, "books": BOOKS, "credits_per_call": 20, "games": games, "snaps": snaps}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "history_snapshots.json"
    plan = build()
    json.dump(plan, open(out, "w"), separators=(",", ":"))
    from collections import Counter
    c = Counter(s[3] for s in plan["snaps"])
    print(out, len(plan["games"]), "games", len(plan["snaps"]), "snapshots", dict(c))
