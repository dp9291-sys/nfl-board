"""Data loading for the NFL ATS model (nflverse sources)."""
import os
import subprocess
import pandas as pd
import numpy as np

DATA_DIR = os.environ.get("NFLATS_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team",
            "away_team", "epa", "success", "pass", "rush", "qb_dropback", "qb_epa", "wp",
            "play_type", "passer_player_id", "passer_player_name", "rusher_player_id",
            "rusher_player_name", "qb_scramble"]


def refresh(seasons, force_current=True):
    """Download games.csv and pbp parquet files. Current season is always re-pulled."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cur = max(seasons)
    subprocess.run(["curl", "-sL", "-o", os.path.join(DATA_DIR, "games.csv"), GAMES_URL], check=True)
    for y in seasons:
        path = os.path.join(DATA_DIR, f"pbp_{y}.parquet")
        if os.path.exists(path) and not (force_current and y == cur):
            continue
        subprocess.run(["curl", "-sL", "-o", path, PBP_URL.format(y=y)], check=True)


def load_games():
    g = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))
    for c in ["home_team", "away_team"]:
        g[c] = g[c].replace(TEAM_MAP)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g


def load_pbp(seasons):
    frames = []
    for y in seasons:
        p = pd.read_parquet(os.path.join(DATA_DIR, f"pbp_{y}.parquet"), columns=PBP_COLS)
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        p[c] = p[c].replace(TEAM_MAP)
    return p


def team_game_stats(pbp):
    """Per (game, offense) efficiency, garbage time removed (WP outside 5-95%)."""
    p = pbp[(pbp["play_type"].isin(["pass", "run"])) & pbp["epa"].notna() & pbp["posteam"].notna()]
    p = p[(p["wp"] >= 0.05) & (p["wp"] <= 0.95)]
    p = p.assign(is_pass=p["qb_dropback"].fillna(0).astype(int))
    p = p.assign(pass_epa=p["epa"] * p["is_pass"])
    agg = p.groupby(["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team"]).agg(
        plays=("epa", "size"), epa=("epa", "sum"), succ=("success", "sum"),
        db=("is_pass", "sum"), pass_epa=("pass_epa", "sum"),
    ).reset_index()
    agg["rush_epa"] = agg["epa"] - agg["pass_epa"]
    agg["home"] = (agg["posteam"] == agg["home_team"]).astype(int)
    return agg


def qb_game_stats(pbp):
    """Per (game, team, passer) dropback EPA, garbage time removed."""
    p = pbp[(pbp["qb_dropback"] == 1) & pbp["qb_epa"].notna() & pbp["passer_player_id"].notna()]
    p = p[(p["wp"] >= 0.05) & (p["wp"] <= 0.95)]
    q = p.groupby(["game_id", "season", "week", "posteam", "passer_player_id"]).agg(
        n=("qb_epa", "size"), epa=("qb_epa", "sum"), name=("passer_player_name", "last")).reset_index()
    return q
