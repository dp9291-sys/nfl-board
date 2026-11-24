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

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--embed-only":
        embed(a[1]); sys.exit(0)
    weekly.run(int(a[0]) if a else None, int(a[1]) if len(a) > 1 else None)
    embed(os.path.join(os.environ["NFLATS_DATA"], "current_week.json"))
    print("index.html rebuilt")
