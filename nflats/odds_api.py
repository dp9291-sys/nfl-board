"""Pull historical NFL spreads/totals from The Odds API and test line-movement signals.

Run on your own machine (the API isn't reachable from Claude's workspace):
    export ODDS_API_KEY=...            # a plan with historical access
    python -m nflats.odds_api fetch 2021 2026      # writes data/odds_history.csv
    python -m nflats.odds_api test                  # adds movement signals and runs the same validation

Snapshots taken per game: "open" = Tuesday 12:00 ET of game week (first stable look-ahead number),
"close" = 10 minutes before kickoff. Books: Pinnacle (sharp, 'eu' region) and US consensus.
Cost: about 40 credits per snapshot call; ~600 calls for 2021-2026.
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from . import data

HIST_PATH = "/v4/historical/sports/americanfootball_nfl/odds"
NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
}


def snapshot(key, when):
    from . import config
    base = config.load()["odds_api_base"]
    q = f"{base}{HIST_PATH}?apiKey={key}&regions=us,eu&markets=spreads,totals&oddsFormat=american&date={when.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    req = urllib.request.Request(q, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            import gzip
            body = gzip.decompress(body)
        return json.loads(body)["data"]


def parse(events, tag):
    rows = []
    for ev in events:
        h, a = NAME_TO_ABBR.get(ev["home_team"]), NAME_TO_ABBR.get(ev["away_team"])
        for bk in ev.get("bookmakers", []):
            rec = {"tag": tag, "home": h, "away": a, "kickoff": ev["commence_time"], "book": bk["key"]}
            for mk in bk["markets"]:
                for o in mk["outcomes"]:
                    if mk["key"] == "spreads" and NAME_TO_ABBR.get(o["name"]) == h:
                        rec["home_spread"], rec["home_spread_price"] = o["point"], o["price"]
                    if mk["key"] == "totals" and o["name"] == "Over":
                        rec["total"] = o["point"]
            rows.append(rec)
    return rows


def fetch(first, last):
    from . import config
    key = os.environ.get("ODDS_API_KEY") or config.load()["odds_api_key"]
    g = data.load_games()
    g = g[(g.season >= first) & (g.season <= last) & g.gametime.notna()]
    rows = []
    for (s, w), wk in g.groupby(["season", "week"]):
        first_day = wk.gameday.min()
        tuesday = first_day - timedelta(days=(first_day.weekday() - 1) % 7 or 7)
        rows += parse(snapshot(key, datetime(tuesday.year, tuesday.month, tuesday.day, 17, tzinfo=timezone.utc)), "open")
        for (day, t), _ in wk.groupby(["gameday", "gametime"]):
            hh, mm = map(int, t.split(":"))
            ko = datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc) + timedelta(hours=5)  # ET -> UTC (approx.)
            rows += parse(snapshot(key, ko - timedelta(minutes=10)), "close")
            time.sleep(0.2)
        print("fetched", s, w, flush=True)
    pd.DataFrame(rows).to_csv(f"{data.DATA_DIR}/odds_history.csv", index=False)


def movement_features():
    o = pd.read_csv(f"{data.DATA_DIR}/odds_history.csv")
    o["date"] = pd.to_datetime(o.kickoff).dt.tz_convert("US/Eastern").dt.date.astype(str)
    agg = o.groupby(["home", "away", "date", "tag"]).agg(spread=("home_spread", "median"), total=("total", "median")).unstack("tag")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    pin = o[o.book == "pinnacle"].groupby(["home", "away", "date", "tag"]).home_spread.median().unstack("tag")
    agg = agg.join(pin.add_prefix("pin_"), how="left").reset_index()
    # Odds API spreads are the home team's handicap (negative = favoured); nflverse spread_line is the opposite sign
    agg["spread_move"] = -(agg.spread_close - agg.spread_open)     # + = moved toward home
    agg["total_move"] = agg.total_close - agg.total_open
    agg["pin_vs_market"] = -(agg.pin_close - agg.spread_close)      # sharp book more home-friendly than consensus
    return agg


def test():
    import pickle
    from . import validate as V
    d, adjs, sig = pickle.load(open(f"{data.DATA_DIR}/v3_signals.pkl", "rb"))
    mv = movement_features()
    d = d.assign(date=d.gameday.dt.date.astype(str)).merge(mv, left_on=["home_team", "away_team", "date"],
                                                            right_on=["home", "away", "date"], how="left")
    fav = np.sign(d.spread_line).replace(0, 1)
    d["total_move_x_fav"] = fav * d.total_move.fillna(0)
    new = {"spread_move": "Spread move, open to close", "total_move_x_fav": "Total move (with favourite)",
           "pin_vs_market": "Pinnacle vs market at close"}
    for c in new:
        d[c] = d[c].fillna(0.0)
    V.CANDIDATES.update(new)
    have = d.spread_open.notna()
    for c, lab in new.items():
        a = V.oos_adjust(d, [c], range(2022, int(d.season.max()) + 1), first=2021)
        print(lab, V.score(d, a, have))


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(int(sys.argv[2]), int(sys.argv[3]))
    else:
        test()
