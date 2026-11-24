"""Plan for the archived-forecast download (python -m nflats.wx_plan OUT.json): every completed outdoor game 2021+,
with stadium coordinates and kickoff time (UTC). Retractable roofs are excluded, matching the live rule."""
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from . import data
from .features import venue_for

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")


def build(first=2021):
    g = data.load_games()
    g = g[(g.season >= first) & g.result.notna() & g.gametime.notna() & (g.roof == "outdoors")]
    games = []
    for r in g.itertuples():
        v = venue_for(r.home_team, r.location, r.stadium, r.roof)
        if not v or not v["outdoor"]:
            continue
        hh, mm = map(int, str(r.gametime).split(":"))
        ko = datetime(r.gameday.year, r.gameday.month, r.gameday.day, hh, mm, tzinfo=ET).astimezone(UTC)
        games.append([r.game_id, int(r.season), round(v["lat"], 3), round(v["lon"], 3), ko.strftime("%Y-%m-%dT%H:%M:%SZ")])
    return {"v": 1, "games": games, "prev_runs_from": "2024-01-01"}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "wx_plan.json"
    p = build()
    json.dump(p, open(out, "w"), separators=(",", ":"))
    from collections import Counter
    print(out, len(p["games"]), "outdoor games", dict(Counter(x[1] for x in p["games"])))
