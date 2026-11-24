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
from nflats.version import MODEL_VERSION

PLAIN_TAG = '<script type="application/json" id="weekdata">__WEEK_JSON__</script>'


def encrypt_payload(data_bytes, lock):
    """AES-256-GCM with a fresh key, key wrapped with the lock's RSA-OAEP public key.
    Only someone with the PIN (which unlocks the private key in the browser) can read it."""
    import base64
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    b64 = lambda b: base64.b64encode(b).decode()
    pub = serialization.load_der_public_key(base64.b64decode(lock["pub"]))
    key, iv = AESGCM.generate_key(256), os.urandom(12)
    ct = AESGCM(key).encrypt(iv, data_bytes, None)
    wk = pub.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return {"v": 1, "iv": b64(iv), "ct": b64(ct), "wk": b64(wk)}


def embed(path_json, out="index.html"):
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = open(os.path.join(here, "template.html"), encoding="utf-8").read()
    raw = open(path_json, encoding="utf-8").read()
    lock_path = os.path.join(here, "lock.json")
    if os.path.exists(lock_path):
        lock = json.load(open(lock_path))
        payload = {"lock": {k: lock[k] for k in ["iter", "salt", "iv", "priv"]}, "enc": encrypt_payload(raw.encode(), lock)}
        html = tpl.replace(PLAIN_TAG, '<script type="application/json" id="weekenc">' + json.dumps(payload) + "</script>")
    else:
        html = tpl.replace("__WEEK_JSON__", raw.replace("<", "\\u003c"))
    open(os.path.join(here, out), "w", encoding="utf-8").write(html)

def gates(wk):
    """Checks before publishing. Returns (failures, warnings). Any failure -> do not publish."""
    import math
    fail, warn = [], []
    games = wk.get("games") or []
    if len(games) < 10:
        fail.append(f"only {len(games)} games")
    c = wk.get("c_st")
    if not isinstance(c, (int, float)) or not (0.3 <= c <= 1.5):
        fail.append(f"special-teams weight out of range: {c}")
    missing_line = [g["game_id"] for g in games if g.get("spread_line") is None]
    if games and len(missing_line) > 0.25 * len(games):
        fail.append(f"{len(missing_line)} of {len(games)} games have no market line")
    elif missing_line:
        warn.append(f"no market line yet: {', '.join(missing_line)}")
    for g in games:
        for k in ("home", "away", "game_id", "st_diff", "h_st", "a_st"):
            v = g.get(k)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                fail.append(f"{g.get('game_id')}: missing {k}")
        sd = g.get("st_diff")
        if isinstance(sd, (int, float)) and isinstance(c, (int, float)) and abs(c * sd) > 5:
            fail.append(f"{g['game_id']}: special-teams adjustment {c * sd:+.1f} pts is implausible")
        sl = g.get("spread_line")
        if isinstance(sl, (int, float)) and abs(sl) > 25:
            fail.append(f"{g['game_id']}: spread {sl} out of range")
    if len({g.get("game_id") for g in games}) != len(games):
        fail.append("duplicate games")
    cv = wk.get("cover2") or {}
    for grp in ("lo", "hi"):
        for band in ("close", "mid", "wide"):
            p = (cv.get(grp) or {}).get(band)
            if not p or len(p.get("w", [])) != 101 or not all(math.isfinite(x) and x > 0 for x in p["w"]):
                fail.append(f"cover model missing or invalid: {grp}/{band}")
    if not (1 <= int(wk.get("week", 0)) <= 22):
        fail.append(f"week out of range: {wk.get('week')}")
    return fail, warn


def log_predictions(wk, here):
    """Append an encrypted, versioned prediction record (skipped when nothing changed since the last one)."""
    import hashlib, datetime
    lock_path = os.path.join(here, "lock.json")
    keep = ["game_id", "home", "away", "gameday", "gametime", "spread_line", "total_line", "st_diff", "h_st", "a_st",
            "pmf_group", "roof"]
    games = [{k: g.get(k) for k in keep} | {"st_adj": round(wk["c_st"] * g["st_diff"], 3),
                                            "stats_line": round(g["intercept"] + sum(g["groups"].values()), 2)}
             for g in wk["games"]]
    body = {"model_version": MODEL_VERSION, "season": wk["season"], "week": wk["week"], "data_through": wk.get("data_through"),
            "c_st": wk["c_st"], "rule": wk.get("rule"), "games": games}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]
    d = os.path.join(here, "predictions")
    os.makedirs(d, exist_ok=True)
    idx_path = os.path.join(d, "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {"files": []}
    same_week = [f for f in idx["files"] if f["season"] == wk["season"] and f["week"] == wk["week"]]
    if same_week and same_week[-1]["hash"] == digest:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    body["built_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = json.dumps(body, default=str).encode()
    name = f"{wk['season']}-w{int(wk['week']):02d}-{now.strftime('%Y%m%dT%H%M%S')}.json"
    if os.path.exists(lock_path):
        payload = {"enc": encrypt_payload(raw, json.load(open(lock_path)))}
    else:
        payload = {"plain": body}
    json.dump(payload, open(os.path.join(d, name), "w"))
    idx["files"].append({"file": name, "season": wk["season"], "week": wk["week"], "hash": digest,
                         "model_version": MODEL_VERSION, "built_at": body["built_at"]})
    json.dump(idx, open(idx_path, "w"), indent=0)
    return name


def finalize(path_json):
    """Stamp version/time, run publishing gates, log the prediction, then embed. Exits 2 on a gate failure."""
    import datetime
    here = os.path.dirname(os.path.abspath(__file__))
    wk = json.load(open(path_json))
    wk["model_version"] = MODEL_VERSION
    wk["built_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fail, warn = gates(wk)
    for w in warn:
        print("GATE WARN:", w)
    if fail:
        for f in fail:
            print("GATE FAIL:", f)
        print("Not publishing: the previous board stays live.")
        sys.exit(2)
    json.dump(wk, open(path_json, "w"), separators=(",", ":"))
    name = log_predictions(wk, here)
    print("prediction logged:", name or "unchanged since last record")
    embed(path_json)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--embed-only":
        embed(a[1]); sys.exit(0)
    if a and a[0] == "--finalize":
        finalize(a[1]); print("index.html rebuilt"); sys.exit(0)
    weekly.run(int(a[0]) if a else None, int(a[1]) if len(a) > 1 else None)
    finalize(os.path.join(os.environ["NFLATS_DATA"], "current_week.json"))
    print("index.html rebuilt")
