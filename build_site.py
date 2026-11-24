"""Rebuild the model and regenerate index.html.

    pip install -r requirements.txt
    python build_site.py            # auto-detects the upcoming week
    python build_site.py 2026 13    # or choose one

Downloads nflverse data into ./data (not committed), runs the full weekly pipeline
(ratings, special teams, injuries, weather if reachable, validation, tiers), then embeds
the week's model into template.html -> index.html.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NFLATS_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
from nflats import weekly

def embed(path_json, out="index.html"):
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = open(os.path.join(here, "template.html"), encoding="utf-8").read()
    data = open(path_json, encoding="utf-8").read().replace("<", "\\u003c")
    open(os.path.join(here, out), "w", encoding="utf-8").write(tpl.replace("__WEEK_JSON__", data))

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--embed-only":
        embed(a[1]); sys.exit(0)
    weekly.run(int(a[0]) if a else None, int(a[1]) if len(a) > 1 else None)
    embed(os.path.join(os.environ["NFLATS_DATA"], "current_week.json"))
    print("index.html rebuilt")
