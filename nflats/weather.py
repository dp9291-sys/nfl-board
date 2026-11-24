"""Kickoff-hour forecasts for outdoor stadiums from Open-Meteo (free, no key).

Averages the three hours from kickoff. Writes data/weather_{season}_{week}.json in the format predict2 reads.
"""
import json
import os
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from . import data, config
from .features import venue_for

INDOOR = {"dome", "closed"}
ET = ZoneInfo("America/New_York")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "nflats/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _phrase(wind, precip_pct, snow, temp):
    bits = []
    if snow and snow > 0.05:
        bits.append("snow")
    elif precip_pct >= 50:
        bits.append("rain likely")
    elif precip_pct >= 25:
        bits.append("chance of rain")
    if wind >= 15:
        bits.append("windy")
    elif wind >= 10:
        bits.append("breezy")
    if temp <= 32:
        bits.append("freezing")
    return ", ".join(bits).capitalize() or "Fair"


def fetch(season, week, log=print):
    cfg = config.load()
    g = data.load_games()
    wk = g[(g.season == season) & (g.week == week)]
    out = {}
    for r in wk.itertuples():
        v = venue_for(r.home_team, r.location, r.stadium, r.roof)
        if not v or not v["outdoor"]:
            continue
        lat, lon = v["lat"], v["lon"]
        try:
            hh, mm = map(int, str(r.gametime).split(":"))
            ko = datetime(r.gameday.year, r.gameday.month, r.gameday.day, hh, mm, tzinfo=ET).astimezone(ZoneInfo("UTC"))
            url = (f"{cfg['weather_base']}/v1/forecast?latitude={lat}&longitude={lon}"
                   "&hourly=temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability,precipitation,snowfall"
                   "&wind_speed_unit=mph&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=UTC&forecast_days=16")
            j = _get(url)["hourly"]
            times = [datetime.fromisoformat(t).replace(tzinfo=ZoneInfo("UTC")) for t in j["time"]]
            idx = [i for i, t in enumerate(times) if ko - timedelta(minutes=30) <= t < ko + timedelta(hours=3)]
            if not idx:
                log(f"  weather: {r.game_id} kickoff beyond forecast range")
                continue
            avg = lambda k: sum((j[k][i] or 0) for i in idx) / len(idx)
            mx = lambda k: max((j[k][i] or 0) for i in idx)
            temp, wind = round(avg("temperature_2m")), round(avg("wind_speed_10m"))
            pp = round(mx("precipitation_probability"))
            snow = round(sum((j["snowfall"][i] or 0) for i in idx) / 2.54, 2)  # cm -> in
            out[r.game_id] = {"temp": temp, "wind": wind, "gust": round(mx("wind_gusts_10m")), "precip_pct": pp,
                              "rain_in": round(sum((j["precipitation"][i] or 0) for i in idx), 2), "snow_in": snow,
                              "phrase": _phrase(wind, pp, snow, temp)}
        except Exception as e:  # keep going; weather is optional
            log(f"  weather: {r.game_id} failed ({e.__class__.__name__}: {e})")
    path = os.path.join(data.DATA_DIR, f"weather_{season}_{week}.json")
    json.dump({"source": "Open-Meteo (3-hour average from kickoff)", "fetched": datetime.now(ET).isoformat(timespec="minutes"),
               "games": out}, open(path, "w"), indent=1)
    log(f"  weather: {len(out)} outdoor games forecast")
    return path
