"""Package / scheme matchup ratings (python -m nflats.packages).

Cells (all available 2016-2026):
  pass plays: formation {shotgun, under center} x defense {blitz (5+ rushers), no blitz}
  run plays : defense box {light <=6, normal 7, heavy 8+}
Box / rushers come from NGS participation (2016-2025) or FTN charting (2022-2026).

For every week, using only earlier plays (decayed: 9-week half-life, prior season x0.55):
  offense : pass rate, shotgun share, EPA per cell (shrunk to league cell mean)
  defense : blitz rate, light/heavy box rates, EPA allowed per cell (shrunk)
Matchup EPA/play for offense O vs defense D
  = pass_rate_O * sum_f sg_O(f) sum_z blitz_D(z) [L(f,z) + oO(f,z) + dD(f,z)]
  + (1 - pass_rate_O) * sum_b box_D(b) [L(b) + oO(b) + dD(b)]
Baseline (pass/run strengths only, no cells) uses the same rates with overall pass and run adjustments.
Interaction = matchup - baseline: what the package/scheme fit adds beyond overall pass/run quality.
Outputs data/packages.parquet with per-game home/away values and home-minus-away differentials.
"""
import os
import sys
import numpy as np
import pandas as pd
from . import data
from .ratings import TEAMS, _weights

PCOLS = ["game_id", "play_id", "season", "week", "posteam", "defteam", "epa", "success", "qb_dropback", "rush",
         "qb_scramble", "shotgun", "wp", "half_seconds_remaining", "down", "play_type"]
K_PASS, K_RUN, K_RATE = 120.0, 80.0, 60.0


def load_plays(first=2015, last=2026):
    frames = []
    for y in range(first, last + 1):
        p = pd.read_parquet(os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"), columns=PCOLS)
        p = p[p.epa.notna() & p.posteam.notna() & p.play_type.isin(["pass", "run"]) & (p.down <= 3)
              & p.wp.between(0.05, 0.95) & (p.half_seconds_remaining > 120)]
        part_p = os.path.join(data.DATA_DIR, f"part_{y}.parquet")
        if os.path.exists(part_p):
            pa = pd.read_parquet(part_p, columns=["nflverse_game_id", "play_id", "defenders_in_box", "number_of_pass_rushers"])
            pa = pa.rename(columns={"nflverse_game_id": "game_id"})
            p = p.merge(pa, on=["game_id", "play_id"], how="left")
        else:
            p["defenders_in_box"] = np.nan
            p["number_of_pass_rushers"] = np.nan
        ftn_p = os.path.join(data.DATA_DIR, f"ftn_{y}.parquet")
        if os.path.exists(ftn_p):
            f = pd.read_parquet(ftn_p, columns=["nflverse_game_id", "nflverse_play_id", "n_defense_box", "n_pass_rushers"])
            f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
            f["play_id"] = pd.to_numeric(f.play_id, errors="coerce")
            p = p.merge(f, on=["game_id", "play_id"], how="left")
            p["defenders_in_box"] = p.defenders_in_box.fillna(p.n_defense_box)
            p["number_of_pass_rushers"] = p.number_of_pass_rushers.fillna(p.n_pass_rushers)
            p = p.drop(columns=["n_defense_box", "n_pass_rushers"])
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    for c in ["posteam", "defteam"]:
        p[c] = p[c].replace(data.TEAM_MAP)
    p["is_pass"] = (p.qb_dropback == 1).astype(int)
    p["sg"] = (p.shotgun == 1).astype(int)
    p["blitz"] = np.where(p.number_of_pass_rushers.notna(), (p.number_of_pass_rushers >= 5).astype(float), np.nan)
    p["box"] = pd.cut(p.defenders_in_box, [-1, 6, 7, 20], labels=["light", "normal", "heavy"]).astype(object)
    p.loc[p.is_pass == 1, "box"] = np.nan
    p.loc[p.is_pass == 0, "blitz"] = np.nan
    return p


PASS_CELLS = [(f, z) for f in (1, 0) for z in (1.0, 0.0)]
RUN_CELLS = ["light", "normal", "heavy"]


def _wsum(df, key, val, w):
    return pd.Series(w * val).groupby(df[key].values).sum(), pd.Series(w).groupby(df[key].values).sum()


def week_ratings(P, season, week, prev_max):
    m = ((P.season == season) & (P.week < week)) | (P.season == season - 1)
    d = P[m]
    if not len(d):
        return None
    w = _weights(d, season, week, prev_max)
    d = d.assign(_w=w)
    out = {}
    pa, ru = d[d.is_pass == 1], d[d.is_pass == 0]
    # league means
    L = {"pass": np.average(pa.epa, weights=pa._w), "run": np.average(ru.epa, weights=ru._w)}
    for f, z in PASS_CELLS:
        c = pa[(pa.sg == f) & (pa.blitz == z)]
        L[(f, z)] = np.average(c.epa, weights=c._w) if len(c) else L["pass"]
    for b in RUN_CELLS:
        c = ru[ru.box == b]
        L[b] = np.average(c.epa, weights=c._w) if len(c) else L["run"]
    out["L"] = L

    def shrunk(df, key, base, K):
        s, n = _wsum(df, key, (df.epa - base).values, df._w.values)
        return np.array((s / (n + K)).reindex(TEAMS).fillna(0.0).values, float)

    def rate(df, key, flag, K, prior):
        s, n = _wsum(df, key, flag.values.astype(float), df._w.values)
        return np.array(((s + K * prior) / (n + K)).reindex(TEAMS).fillna(prior).values, float)

    # overall pass/run adjustments, then each team's cell residuals relative to its own overall level
    for side, key in (("o", "posteam"), ("d", "defteam")):
        out[f"{side}_pass"] = shrunk(pa, key, L["pass"], K_PASS)
        out[f"{side}_run"] = shrunk(ru, key, L["run"], K_RUN)
    # cell residuals: each play's EPA vs its cell's league mean, minus the team's own raw average of that
    # quantity in the phase (pass or run), so residuals average ~0 for every team and carry no overall level
    pa_t = pa[pa.blitz.notna()].copy()
    pa_t["x"] = pa_t.epa.values - np.array([L[(f, z)] for f, z in zip(pa_t.sg, pa_t.blitz)])
    ru_t = ru[ru.box.notna()].copy()
    ru_t["x"] = ru_t.epa.values - ru_t.box.map({b: L[b] for b in RUN_CELLS}).values
    for side, key in (("o", "posteam"), ("d", "defteam")):
        for df, cells, kind, K in ((pa_t, PASS_CELLS, "p", K_PASS), (ru_t, RUN_CELLS, "r", K_RUN)):
            if not len(df):
                for c in cells:
                    out[f"{side}_{kind}{c[0]}{int(c[1])}" if kind == "p" else f"{side}_r{c}"] = np.zeros(len(TEAMS))
                continue
            sw = pd.Series(df._w.values * df.x.values).groupby(df[key].values).sum()
            nw = pd.Series(df._w.values).groupby(df[key].values).sum()
            lvl = (sw / nw).to_dict()
            r = df.x.values - df[key].map(lvl).values
            df = df.assign(_r=r)
            for c in cells:
                cc = df[(df.sg == c[0]) & (df.blitz == c[1])] if kind == "p" else df[df.box == c]
                name = f"{side}_p{c[0]}{int(c[1])}" if kind == "p" else f"{side}_r{c}"
                s_, n_ = _wsum(cc, key, cc._r.values, cc._w.values)
                out[name] = np.array((s_ / (n_ + K)).reindex(TEAMS).fillna(0.0).values, float)
    allp = d
    lg_pass_rate = np.average(allp.is_pass, weights=allp._w)
    out["pass_rate"] = rate(allp, "posteam", allp.is_pass, K_RATE * 3, lg_pass_rate)
    out["sg_rate"] = rate(pa, "posteam", pa.sg, K_RATE, np.average(pa.sg, weights=pa._w))
    pb = pa[pa.blitz.notna()]
    out["blitz_rate"] = rate(pb, "defteam", pb.blitz, K_RATE, np.average(pb.blitz, weights=pb._w)) if len(pb) else np.full(len(TEAMS), 0.25)
    rb = ru[ru.box.notna()]
    for b in RUN_CELLS:
        out[f"box_{b}"] = rate(rb, "defteam", (rb.box == b), K_RATE, np.average(rb.box == b, weights=rb._w)) if len(rb) else np.full(len(TEAMS), 1 / 3)
    tot = out["box_light"] + out["box_normal"] + out["box_heavy"]
    for b in RUN_CELLS:
        out[f"box_{b}"] = np.asarray(out[f"box_{b}"], float) / tot
    return out


def matchup(R, o, dfn):
    """Expected EPA/play for offense o vs defense dfn: (full, baseline). full = baseline + interaction, where the
    interaction weights each team's own cell residuals by this opponent's tendencies."""
    L = R["L"]
    pr, sg, bz = R["pass_rate"][o], R["sg_rate"][o], R["blitz_rate"][dfn]
    base = pr * (L["pass"] + R["o_pass"][o] + R["d_pass"][dfn]) + (1 - pr) * (L["run"] + R["o_run"][o] + R["d_run"][dfn])
    ip = 0.0
    for f, z in PASS_CELLS:
        wf = sg if f == 1 else 1 - sg
        wz = bz if z == 1.0 else 1 - bz
        ip += wf * wz * (R[f"o_p{f}{int(z)}"][o] + R[f"d_p{f}{int(z)}"][dfn])
    ir = sum(R[f"box_{b}"][dfn] * (R[f"o_r{b}"][o] + R[f"d_r{b}"][dfn]) for b in RUN_CELLS)
    inter = pr * ip + (1 - pr) * ir
    return base + inter, base


def build(last=2026):
    P = load_plays(2015, last)
    print("plays", len(P), "| box tagged", round(P[P.is_pass == 0].box.notna().mean(), 2),
          "| blitz tagged", round(P[P.is_pass == 1].blitz.notna().mean(), 2), flush=True)
    g = data.load_games()
    g = g[(g.season >= 2016) & (g.season <= last)]
    rows = []
    idx = {t: i for i, t in enumerate(TEAMS)}
    for (S, W), wk in g.sort_values(["season", "week"]).groupby(["season", "week"]):
        prev_max = P.loc[P.season == S - 1, "week"].max()
        R = week_ratings(P, S, W, prev_max if prev_max == prev_max else 18)
        if R is None:
            continue
        for r in wk.itertuples():
            if r.home_team not in idx or r.away_team not in idx:
                continue
            h, a = idx[r.home_team], idx[r.away_team]
            fh, bh = matchup(R, h, a)   # home offense vs away defense
            fa, ba = matchup(R, a, h)
            rows.append({"game_id": r.game_id, "season": S, "week": W,
                         "h_match": fh, "a_match": fa, "h_base": bh, "a_base": ba,
                         "match_diff": fh - fa, "base_diff": bh - ba, "inter_diff": (fh - bh) - (fa - ba),
                         "h_inter": fh - bh, "a_inter": fa - ba})
        if W == 1:
            print("  packages", S, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(data.DATA_DIR, "packages.parquet"))
    return out


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
