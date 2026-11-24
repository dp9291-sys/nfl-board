"""Margin model, walk-forward backtest, and cover probabilities."""
import numpy as np
import pandas as pd

POWER_FEATS = ["net_epa_diff", "net_sr_diff", "margin_diff", "qb_adj_diff", "hfa_flag",
               "rest_diff", "home_bye", "away_bye", "home_short", "away_short",
               "tz_shift", "early_west", "div"]
# Tested and dropped: away_travel_k (absorbed weak LA/LV home fields -> spurious), night_west_east
# (tiny sample, sign opposite the published body-clock research). Both computed in features.py.
SIGMA = 13.3


SEASON_HL = 4.0  # seasons; recent eras (lower HFA, more passing) count more


def season_w(seasons, target):
    return 0.5 ** ((target - 1 - np.asarray(seasons)) / SEASON_HL)


def ridge_fit(X, y, lam=None, unpen=(0,), w=None):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.ones(len(X)) if w is None else np.asarray(w) / np.mean(w)
    lam = lam if lam is not None else 1.0
    pen = np.full(X.shape[1], lam * len(X) / 1000.0)
    for u in unpen:
        pen[u] = 1e-8
    b = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(pen), X.T @ (w * y))
    return b


def ridge_pred(b, X):
    return b[0] + np.asarray(X) @ b[1:]


def _std(train, test, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    return (train[cols] - mu) / sd, (test[cols] - mu) / sd


def walk_forward(df, test_seasons, first_train=2010, blend_first=2013, lam=20.0):
    """Returns test-set predictions for each model variant (no look-ahead)."""
    df = df[df.result.notna() & df.spread_line.notna()].copy()
    # Stage A out-of-sample power projections for every season >= blend_first
    df["projA"] = np.nan
    for s in sorted(df.season.unique()):
        if s < blend_first:
            continue
        tr = df[(df.season >= first_train) & (df.season < s)]
        te = df.season == s
        Xtr, Xte = _std(tr, df[te], POWER_FEATS)
        b = ridge_fit(Xtr.values, tr.result.clip(-28, 28).values, lam=lam, w=season_w(tr.season, s))
        df.loc[te, "projA"] = ridge_pred(b, Xte.values)
    out = []
    for S in test_seasons:
        te = df[df.season == S].copy()
        tr = df[(df.season >= blend_first) & (df.season < S)]
        # Blend: residual-vs-market regression  ats = c + k*(projA - spread)
        x = (tr.projA - tr.spread_line).values
        y = (tr.result - tr.spread_line).clip(-28, 28).values
        k = np.polyfit(x, y, 1)
        te["pred_blend"] = te.spread_line + np.polyval(k, (te.projA - te.spread_line).values)
        te["k_slope"], te["k_int"] = k[0], k[1]
        # Single-stage: result ~ spread + features
        cols = ["spread_line"] + POWER_FEATS
        tr2 = df[(df.season >= first_train) & (df.season < S)]
        Xtr, Xte = _std(tr2, te, cols)
        b = ridge_fit(Xtr.values, (tr2.result).clip(-28, 28).values, lam=lam)
        te["pred_single"] = ridge_pred(b, Xte.values)
        out.append(te)
    return pd.concat(out)


def ats_eval(te, pred_col, thresholds=(0, 1, 1.5, 2, 2.5, 3, 4)):
    e = te.copy()
    edge = e[pred_col] - e.spread_line  # >0 => take home
    ats = e.result - e.spread_line
    rows = []
    for t in thresholds:
        m = (edge.abs() > t) & (ats != 0)
        win = np.sign(edge[m]) == np.sign(ats[m])
        n = int(m.sum())
        w = int(win.sum())
        rows.append({"model": pred_col, "edge_gt": t, "bets": n, "wins": w,
                     "hit%": round(100 * w / n, 1) if n else np.nan,
                     "roi%_-110": round(100 * (w * (100 / 110) - (n - w)) / n, 1) if n else np.nan})
    return pd.DataFrame(rows)


# ---------- Cover probability with key numbers ----------

def margin_pmf_table(df):
    """Empirical P(result = k | spread bucket). Captures key numbers 3, 7, 10, 6, 4, 14."""
    d = df[df.result.notna() & df.spread_line.notna()]
    ks = np.arange(-60, 61)
    table = {}
    for s in np.arange(-15, 15.5, 0.5):
        near = d[(d.spread_line - s).abs() <= 1.5]
        resid = (near.result - near.spread_line).values  # residual relative to line
        table[s] = resid
    return table, ks


def cover_prob(mu, line, d_hist):
    """P(home covers), P(push) at a given home line (home -line is spread_line>0 means home favored).

    Uses residuals from historical games with similar spreads, shifted to the projected mean,
    so the lumpy distribution around 3 and 7 is preserved.
    """
    s = np.clip(np.round(mu * 2) / 2, -15, 15)
    near = d_hist[(d_hist.spread_line - s).abs() <= 1.5]
    sim = near.result.values - near.spread_line.values + mu
    # Re-integerize so pushes are possible, honouring empirical key-number mass
    sim = np.round(sim - (mu - s))  # residual around the bucket's own integer structure
    sim = sim + (mu - s)
    # Smooth: mix with normal for robustness
    from scipy.stats import norm
    p_emp_win = np.mean(sim > line)
    p_emp_push = np.mean(np.isclose(sim, line)) if float(line).is_integer() else 0.0
    p_norm_win = 1 - norm.cdf(line + (0.5 if float(line).is_integer() else 0), mu, SIGMA)
    p_norm_push = (norm.cdf(line + 0.5, mu, SIGMA) - norm.cdf(line - 0.5, mu, SIGMA)) if float(line).is_integer() else 0.0
    return 0.6 * p_emp_win + 0.4 * p_norm_win, 0.6 * p_emp_push + 0.4 * p_norm_push


# ---------- Key-number-aware cover probability (exponential tilting) ----------
KS = np.arange(-50, 51)


def pmf_tables(df, max_season):
    """Empirical integer-margin pmf for each half-point spread bucket (games before max_season)."""
    d = df[df.result.notna() & df.spread_line.notna() & (df.season < max_season)]
    tabs = {}
    for s in np.arange(-20, 20.5, 0.5):
        win = 0.25
        near = d[(d.spread_line - s).abs() <= win]
        while len(near) < 120 and win < 10:
            win += 0.5
            near = d[(d.spread_line - s).abs() <= win]
        # raw final margins: key numbers (3, 7, 10...) are absolute, so do NOT recentre
        k = np.clip(near.result.round(), -50, 50).astype(int)
        cnt = np.bincount(k + 50, minlength=101).astype(float)
        sm = cnt + 0.02 * cnt.sum() / 101  # small floor so tails are never zero
        tabs[float(s)] = sm / sm.sum()
    return tabs


def pmf_at(mu, tabs):
    """Result pmf for a game whose fair line is mu: interpolate buckets, mirror for symmetry."""
    mu = float(np.clip(mu, -20, 20))
    lo = np.floor(mu * 2) / 2
    hi = min(lo + 0.5, 20.0)
    f = 0 if hi == lo else (mu - lo) / 0.5
    def sym(b):
        return 0.5 * (tabs[b] + tabs[-b][::-1])
    return (1 - f) * sym(lo) + f * sym(hi)


def cover_probs(mu, line, tabs):
    """Home perspective. mu = projected home margin; line = nflverse spread_line
    (expected home margin; home covers if result > line). Returns (P home, P push, P away).

    P(home) = neutral 50/50 split of non-push mass + [P_mu(>line) - P_line(>line)],
    so a fair line is always a coin flip and only the projection gap creates an edge.
    """
    p_mu, p_ln = pmf_at(mu, tabs), pmf_at(line, tabs)
    push = p_mu[KS == line].sum()
    push0 = p_ln[KS == line].sum()
    delta = p_mu[KS > line].sum() - p_ln[KS > line].sum()
    home = (1 - push0) / 2 + delta
    away = 1 - home - push
    return home, push, away


TOTAL_SPLIT = 43.5  # low-total games land on 3 far more often (20.6% vs 14.6% at a 3-point line)


def pmf_tables_by_total(df, max_season):
    d = df[df.result.notna() & df.spread_line.notna() & (df.season < max_season) & df.total_line.notna()]
    return {"lo": pmf_tables(d[d.total_line <= TOTAL_SPLIT], max_season),
            "hi": pmf_tables(d[d.total_line > TOTAL_SPLIT], max_season)}


def effective_total(total_line, wind_mph):
    """Closing totals still ran ~1.3-2 pts high in 10-19 mph wind (2013-2026), so shade them."""
    if total_line is None or np.isnan(total_line):
        return 44.0
    return total_line - (1.5 if (wind_mph or 0) >= 10 else 0.0)
