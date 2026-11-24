"""Smooth key-number cover model.

P(final home margin = k | fair line mu) is proportional to  base(k - mu) * w_k
  - base: mix of a narrow and a wide normal curve (close games + blowouts), shifts with the projection
  - w_k: key-number weights for each exact margin (3, 7, 10, 6, 4, 14 ... get more mass, 0 and 1-pt games less),
         symmetric (w_k = w_-k), fitted by maximum likelihood on past games only, shrunk toward 1
  - fitted separately for low-total (<= 43.5) and normal-total games, and by how close the market expects
    the game to be (|line| <= 4.5, 5-9.5, 10+): exact 3s are far more common in close games. The band comes
    from the market line, not the projection, so for any one game the shape is fixed and cover odds are smooth.
Because the shape is a smooth shift with fixed weights, cover odds always move in the right
direction as the projection moves (monotone likelihood ratio), and nearby lines can't jump around.
Cover odds use the same anchoring as before: a projection equal to the line is a coin flip.
"""
import numpy as np
import pandas as pd
from . import data

KS = np.arange(-50, 51)
TOTAL_SPLIT = 43.5


def base(x, sh):
    s1, s2, p = sh
    return p * np.exp(-0.5 * (x / s1) ** 2) / s1 + (1 - p) * np.exp(-0.5 * (x / s2) ** 2) / s2


def _fit_group(lines, results, wts, sh, lam=3.0):
    """Penalized maximum likelihood for symmetric log key weights u_|k| (shrunk toward 0, i.e. w = 1)."""
    from scipy.optimize import minimize
    k = np.clip(np.round(results), -50, 50).astype(int) + 50
    phi = base(KS[None, :] - lines[:, None], sh)
    cnt = np.bincount(k, weights=wts, minlength=101)
    absk = np.abs(KS)

    def f(u):
        v = u[absk]
        e = phi * np.exp(v)[None, :]
        Z = e.sum(1)
        ll = float(np.sum(wts * (np.log(phi[np.arange(len(k)), k]) + v[k] - np.log(Z))))
        E = (wts[:, None] * e / Z[:, None]).sum(0)
        g = np.bincount(absk, weights=cnt - E, minlength=51)
        return -(ll - lam * np.sum(u * u)), -(g - 2 * lam * u)

    r = minimize(f, np.zeros(51), jac=True, method="L-BFGS-B")
    u = r.x
    w = np.exp(u[absk])
    w /= np.average(w, weights=phi.sum(0))
    return w, -r.fun


SHAPES = [(s1, s2, p) for s1 in (5.0, 7.0, 9.0) for s2 in (13.0, 14.5, 16.0) for p in (0.1, 0.2, 0.3, 0.4)]


def fit(games, max_season, first=2002, half_life=10.0, shapes=SHAPES):
    g = games[games.result.notna() & games.spread_line.notna() & games.total_line.notna()
              & (games.season < max_season) & (games.season >= first)]
    out = {}
    a = g.spread_line.abs()
    for grp, m in (("lo", g.total_line <= TOTAL_SPLIT), ("hi", g.total_line > TOTAL_SPLIT)):
        out[grp] = {}
        for band, bm in (("close", a <= 4.5), ("mid", (a > 4.5) & (a < 10)), ("wide", a >= 10)):
            x = g[m & bm]
            wts = 0.5 ** ((max_season - 1 - x.season.values) / half_life)
            best = None
            for sh in shapes:
                w, ll = _fit_group(x.spread_line.values.astype(float), x.result.values.astype(float), wts, sh)
                if best is None or ll > best[2]:
                    best = (sh, w, ll)
            out[grp][band] = {"shape": [float(v) for v in best[0]], "w": best[1]}
    return out


def band(market_line):
    a = abs(market_line)
    return "close" if a <= 4.5 else "mid" if a < 10 else "wide"


def params(par, grp, market_line):
    return par[grp][band(market_line)]


def pmf(mu, par):
    p = base(KS - mu, par["shape"]) * par["w"]
    return p / p.sum()


def cover_probs(mu, line, par):
    p_mu, p_ln = pmf(mu, par), pmf(line, par)
    push = p_mu[KS == line].sum()
    push0 = p_ln[KS == line].sum()
    delta = p_mu[KS > line].sum() - p_ln[KS > line].sum()
    home = (1 - push0) / 2 + delta
    return home, push, 1 - home - push


def group(total_line, wind_mph):
    et = 44.0 if total_line is None or total_line != total_line else total_line - (1.5 if (wind_mph or 0) >= 10 else 0.0)
    return "lo" if et <= TOTAL_SPLIT else "hi"


def export(par):
    return {g: {b: {"shape": [round(v, 3) for v in par[g][b]["shape"]], "w": [round(float(v), 5) for v in par[g][b]["w"]]}
                for b in par[g]} for g in par}


RULE = {"thr": 0.5, "edge": 0.02, "cap": 0.02}


def enrich(path):
    """Add the smooth cover model (fit on all completed games) and the betting rule to the week file."""
    import json
    wk = json.load(open(path))
    wk["cover2"] = export(fit(data.load_games(), int(wk["season"]) + 1))
    wk["rule"] = RULE
    json.dump(wk, open(path, "w"), separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return wk
