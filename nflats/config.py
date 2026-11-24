"""Local settings (your Odds API key lives here, on your machine only)."""
import json
import os
from .data import DATA_DIR

PATH = os.path.join(DATA_DIR, "config.json")
DEFAULTS = {"odds_api_key": "", "poll_minutes": 60, "cache_minutes": 10,
            "regions": "us,eu", "odds_api_base": "https://api.the-odds-api.com",
            "weather_base": "https://api.open-meteo.com"}


def _file():
    cfg = dict(DEFAULTS)
    try:
        with open(PATH) as f:
            cfg.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return cfg


def load():
    cfg = _file()
    # environment overrides (handy for testing / servers)
    if os.environ.get("ODDS_API_KEY"):
        cfg["odds_api_key"] = os.environ["ODDS_API_KEY"]
    for k, env in [("odds_api_base", "ODDS_API_BASE"), ("weather_base", "WEATHER_BASE")]:
        if os.environ.get(env):
            cfg[k] = os.environ[env]
    return cfg


def save(**updates):
    cfg = _file()  # never persist environment overrides
    cfg.update({k: v for k, v in updates.items() if v is not None})
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: cfg[k] for k in DEFAULTS}, f, indent=2)
    os.replace(tmp, PATH)
    try:
        os.chmod(PATH, 0o600)
    except OSError:
        pass
    return cfg
