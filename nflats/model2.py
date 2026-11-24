"""v2 model: feature-group ablation with walk-forward validation, and ATS-residual model."""
import numpy as np
import pandas as pd
from .model import ridge_fit, ridge_pred, season_w, POWER_FEATS

INJ = [f"inj_{g}_diff" for g in ["WR", "TE", "RB", "OL", "DL", "LB", "DB"]]
GROUPS = {
    "base (v1)": [],
    "+ early-down / turnover-free / pass EPA": ["early_diff", "nto_diff", "pass_diff"],
    "+ explosives & sack rates": ["expl_diff", "sack_diff"],
    "+ special teams": ["st_diff"],
    "+ turnovers, PROE, penalties": ["to_diff", "fum_diff", "proe_diff", "pen_diff"],
    "+ QB CPOE": ["qb_cpoe_diff"],
    "+ injuries (snap-weighted)": INJ,
    "+ weather": ["cold_away_warm", "cold_home_warm", "wx_compress"],
}
MARKET = ["mkt_dev", "last_ats_diff", "wind_x_spread"]


def prep(f):
    f = f.copy()
    f["wx_compress"] = f.wind15 * f.margin_diff
    f["wind_x_spread"] = f.wind15 * f.spread_line
    return f[f.result.notna() & f.spread_line.notna()]


def _std(tr, te, cols):
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
    return (tr[cols] - mu) / sd, (te[cols] - mu) / sd


def project(df, feats, first=2013, lam=20.0):
    """Out-of-sample power projections for every season > first (train on prior seasons)."""
    out = pd.Series(np.nan, index=df.index)
    for s in sorted(df.season.unique()):
        if s <= first + 1:
            continue
        tr, te = df[(df.season >= first) & (df.season < s)], df.season == s
        Xtr, Xte = _std(tr, df[te], feats)
        b = ridge_fit(Xtr.values, tr.result.clip(-28, 28).values, lam=lam, w=season_w(tr.season, s))
        out[te] = ridge_pred(b, Xte.values)
    return out


def ats_stats(df, pred, mask=None, t=(0, 1, 2, 3)):
    e, a = pred - df.spread_line, df.result - df.spread_line
    r = {}
    for th in t:
        m = (e.abs() > th) & (a != 0) & e.notna()
        if mask is not None:
            m &= mask
        r[f"hit>{th}"] = round(100 * float((np.sign(e[m]) == np.sign(a[m])).mean()), 1)
        r[f"n>{th}"] = int(m.sum())
    return r


def ablation(f, test_from=2018):
    df = prep(f)
    feats = list(POWER_FEATS)
    rows, projs = [], {}
    test = df.season >= test_from
    for name, add in GROUPS.items():
        feats = feats + add
        p = project(df, feats)
        projs[name] = p
        r = {"model": name, "n_feats": len(feats),
             "MAE": round(float((df.result[test] - p[test]).abs().mean()), 3)}
        r.update(ats_stats(df[test], p[test]))
        rows.append(r)
    r = {"model": "market closing line", "n_feats": 0,
         "MAE": round(float((df.result[test] - df.spread_line[test]).abs().mean()), 3)}
    rows.append(r)
    return pd.DataFrame(rows), projs, feats, df


def ats_model(df, proj, extra, test_from=2018, lam=50.0):
    """Predict ATS residual (result - close) from (proj - close) + extra market features, walk-forward."""
    d = df.assign(gap=proj - df.spread_line, ats=(df.result - df.spread_line).clip(-21, 21))
    cols = ["gap"] + extra
    pred = pd.Series(np.nan, index=d.index)
    coefs = {}
    for s in range(test_from, int(d.season.max()) + 1):
        tr = d[(d.season < s) & d.gap.notna()]
        te = d.season == s
        Xtr, Xte = _std(tr, d[te], cols)
        b = ridge_fit(Xtr.values, tr.ats.values, lam=lam, w=season_w(tr.season, s))
        pred[te] = d.spread_line[te] + ridge_pred(b, Xte.values)
        coefs[s] = dict(zip(["icpt"] + cols, np.round(np.r_[b[0], b[1:] / tr[cols].std().values], 4)))
    return pred, coefs
