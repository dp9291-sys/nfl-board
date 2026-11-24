"""Compare the current bucket pmf with kernel-smoothed pmfs (walk-forward 2018-2026)."""
import pickle
import sys
import numpy as np
import pandas as pd
from . import data, model as M, validate as V

KS = M.KS
PAY = lambda o: o / 100 if o > 0 else 100 / abs(o)


def tables_smooth(df, max_season, bw):
    d = df[df.result.notna() & df.spread_line.notna() & (df.season < max_season)]
    sl = d.spread_line.values
    k = np.clip(d.result.round(), -50, 50).astype(int).values + 50
    tabs = {}
    for s in np.arange(-20, 20.5, 0.5):
        w = np.exp(-0.5 * ((sl - s) / bw) ** 2)
        cnt = np.bincount(k, weights=w, minlength=101)
        sm = cnt + 0.02 * cnt.sum() / 101
        tabs[float(s)] = sm / sm.sum()
    return tabs


def by_total(df, S, maker):
    d = df[df.total_line.notna()]
    return {"lo": maker(d[d.total_line <= M.TOTAL_SPLIT], S), "hi": maker(d[d.total_line > M.TOTAL_SPLIT], S)}


def monotone_violations(T):
    n = bad = 0
    worst = 0.0
    for grp in ("lo", "hi"):
        for L in np.arange(-14, 14.5, 0.5):
            prev = None
            for mu in np.arange(L - 4, L + 4.01, 0.25):
                h = M.cover_probs(mu, L, T[grp])[0]
                if prev is not None:
                    n += 1
                    if h < prev - 1e-9:
                        bad += 1
                        worst = max(worst, prev - h)
                prev = h
    return bad, n, worst


def run(d, makers, seasons=range(2018, 2027)):
    out = {}
    for name, mk in makers.items():
        rows, mono = [], None
        for S in seasons:
            T = by_total(d, S, mk)
            if S == 2026:
                mono = monotone_violations(T)
            X = d[(d.season == S) & d.result.notna() & d.spread_line.notna() & d.st_adj.notna()]
            for r in X.itertuples():
                tab = T["lo" if M.effective_total(r.total_line, r.wind_mph) <= M.TOTAL_SPLIT else "hi"]
                mu = r.spread_line + r.st_adj
                h, p, a = M.cover_probs(mu, r.spread_line, tab)
                oh = r.home_spread_odds if r.home_spread_odds == r.home_spread_odds else -110
                oa = r.away_spread_odds if r.away_spread_odds == r.away_spread_odds else -110
                evh, eva = h * PAY(oh) - a, a * PAY(oa) - h
                res = r.result - r.spread_line
                home_cov = np.nan if res == 0 else float(res > 0)
                if evh >= eva:
                    rows.append((S, evh, res, oh, h / (h + a), home_cov))
                else:
                    rows.append((S, eva, -res, oa, a / (h + a), home_cov))
        E = pd.DataFrame(rows, columns=["S", "ev", "res", "odds", "p_nopush", "home_cov"])
        out[name] = (E, mono)
    return out


def tier_stats(E, lo, hi):
    e = E[(E.ev >= lo) & (E.ev < hi)]
    pnl = np.where(e.res > 0, [PAY(o) for o in e.odds], np.where(e.res < 0, -1.0, 0.0))
    dec = e[e.res != 0]
    return {"n": len(e), "win": round(100 * float((dec.res > 0).mean()), 1) if len(dec) else None,
            "roi": round(100 * float(pnl.mean()), 1) if len(e) else None,
            "pred": round(100 * float(dec.p_nopush.mean()), 1) if len(dec) else None}


def main():
    d, _, _ = pickle.load(open(f"{data.DATA_DIR}/v3_signals.pkl", "rb"))
    tests = [s for s in sorted(d.season.unique()) if s >= 2016]
    d = d.assign(st_adj=V.oos_adjust(d, ["st_diff"], tests))
    makers = {"current buckets": M.pmf_tables,
              "smoothed bw=1.0": lambda df, S: tables_smooth(df, S, 1.0),
              "smoothed bw=1.5": lambda df, S: tables_smooth(df, S, 1.5),
              "smoothed bw=2.5": lambda df, S: tables_smooth(df, S, 2.5)}
    res = run(d, makers)
    for name, (E, mono) in res.items():
        dec = E[E.res != 0]
        brier = float(np.mean((dec.p_nopush - (dec.res > 0)) ** 2))
        print(f"\n== {name}: wrong-direction steps {mono[0]}/{mono[1]} (worst {mono[2]:.4f}); picked-side Brier {brier:.4f}")
        for lab, lo, hi in [("Tier A", .05, 9), ("Tier B", .03, .05), ("Pass", -9, .03)]:
            print(f"   {lab}: {tier_stats(E, lo, hi)}")
        # calibration of the picked side's cover probability (no-push), bins
        dec = dec.assign(b=pd.cut(dec.p_nopush, [0, .5, .52, .54, .56, .58, 1]))
        cal = dec.groupby("b", observed=True).apply(lambda g: (len(g), round(100 * g.p_nopush.mean(), 1), round(100 * (g.res > 0).mean(), 1)))
        print("   calibration (n, predicted %, actual %):", {str(k): v for k, v in cal.items()})
        print("   Tier A by era:", {f"{a}-{b}": tier_stats(E[(E.S >= a) & (E.S <= b)], .05, 9)["roi"] for a, b in [(2018, 2020), (2021, 2023), (2024, 2026)]})


if __name__ == "__main__":
    main()
