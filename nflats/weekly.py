"""One-command weekly rebuild:  python -m nflats.weekly  [season week]

Downloads fresh nflverse data, rebuilds every rating and feature walk-forward, fetches weather,
reruns the validation, and writes data/current_week.json for the app.
"""
import os
import shutil
import subprocess
import sys
import pickle
import warnings
import pandas as pd
from . import data

warnings.filterwarnings("ignore")
REL = "https://github.com/nflverse/nflverse-data/releases/download"


def _dl(url, path, log, force=False):
    if os.path.exists(path) and not force:
        return
    tmp = path + ".part"
    r = subprocess.run(["curl", "-sfL", "-o", tmp, url])
    if r.returncode != 0:
        import urllib.request
        urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    log(f"  downloaded {os.path.basename(path)}")


def detect_week():
    g = data.load_games()
    s = int(g[g.result.notna()].season.max())
    cur = g[g.season == s]
    for w in sorted(cur[cur.result.isna()].week.unique()):
        wk = cur[cur.week == w]
        if wk.result.isna().mean() > 0.5:  # a lone Monday-night game doesn't hold the board on last week
            return s, int(w)
    return s + 1, 1  # season over


def refresh_data(season, log=print):
    os.makedirs(data.DATA_DIR, exist_ok=True)
    _dl(data.GAMES_URL, os.path.join(data.DATA_DIR, "games.csv"), log, force=True)
    for y in range(2009, season + 1):
        _dl(f"{REL}/pbp/play_by_play_{y}.parquet", os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"), log, force=(y == season))
    for y in range(2012, season + 1):
        _dl(f"{REL}/injuries/injuries_{y}.parquet", os.path.join(data.DATA_DIR, f"inj_{y}.parquet"), log, force=(y == season))
        _dl(f"{REL}/snap_counts/snap_counts_{y}.parquet", os.path.join(data.DATA_DIR, f"snap_{y}.parquet"), log, force=(y == season))
    _dl(f"{REL}/players/players.parquet", os.path.join(data.DATA_DIR, "players.parquet"), log, force=True)


def run(season=None, week=None, log=print, download=True):
    if download:
        log("Step 1/8  Downloading latest nflverse data")
        if not os.path.exists(os.path.join(data.DATA_DIR, "games.csv")):
            _dl(data.GAMES_URL, os.path.join(data.DATA_DIR, "games.csv"), log, force=True)
        s0, _ = detect_week()
        refresh_data(season or s0, log)
    if season is None or week is None:
        season, week = detect_week()
    log(f"Building week {week} of {season}")
    from . import build, predict, build2, predict2, build3, validate, predict3, weather
    log("Step 2/8  Team ratings and QB values (walk-forward)")
    build.build(season, verbose=False).to_parquet(f"{data.DATA_DIR}/features.parquet")
    log("Step 3/8  Base projections")
    predict.main(season, week)
    log("Step 4/8  Secondary stats, special teams, injuries")
    build2.build2(season).to_parquet(f"{data.DATA_DIR}/features2.parquet")
    log("Step 5/8  Weather forecasts (Open-Meteo)")
    try:
        wx = weather.fetch(season, week, log)
    except Exception as e:
        log(f"  weather unavailable ({e}); continuing without it")
        wx = None
    log("Step 6/8  Full stats line and key-number tables")
    predict2.main(season, week, wx)
    log("Step 7/8  Candidate signals and validation")
    f3 = build3.build3(season)
    f3.to_parquet(f"{data.DATA_DIR}/features3.parquet")
    d = validate.prep(f3)
    t, adjs = validate.single_signals(d)
    pickle.dump((d, adjs, t), open(f"{data.DATA_DIR}/v3_signals.pkl", "wb"))
    log("Step 8/8  Confidence tiers and this week's board")
    path, _ = predict3.main(season, week)
    shutil.copy(path, os.path.join(data.DATA_DIR, "current_week.json"))
    log(f"Done: week {week} of {season} is ready")
    return season, week


if __name__ == "__main__":
    a = sys.argv[1:]
    run(int(a[0]) if a else None, int(a[1]) if len(a) > 1 else None)
