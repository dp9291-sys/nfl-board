"""v4 player-level and matchup signals (python -m nflats.build4 2026). All lookahead-free.

  unit_match      : each offense's pass and run units vs the opposing defense's pass and run units,
                    weighted by how often that offense passes (EPA per play x 62 plays)
  deep_match      : deep (20+ air yards) passing offense vs opposing deep pass defense, weighted by deep rate
  off_form/def_form: last-4-game EPA vs season-long rating (is a unit trending)
  avail_*         : production-weighted value of players listed Out/Doubtful/Questionable:
                    receivers (targets x EPA/target vs replacement), backs (carries x EPA/carry),
                    pass rushers (sacks/hits), coverage players (INTs/breakups)
  qb_blitz_match  : QB's EPA when blitzed vs not (FTN charting) x opposing defense blitz rate (2022+)
"""
import os
import sys
import numpy as np
import pandas as pd
from . import data, advanced as A
from .ratings import TEAMS, TIDX, _weights

FIRST = 2013
COLS = ["game_id", "play_id", "season", "week", "posteam", "defteam", "home_team", "epa", "wp", "play_type",
        "qb_dropback", "air_yards", "receiver_player_id", "rusher_player_id", "passer_player_id", "qb_epa",
        "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id", "qb_hit_1_player_id", "qb_hit_2_player_id",
        "interception_player_id", "pass_defense_1_player_id", "pass_defense_2_player_id"]


def load(last):
    fr = [pd.read_parquet(os.path.join(data.DATA_DIR, f"pbp_{y}.parquet"), columns=COLS) for y in range(FIRST - 1, last + 1)]
    p = pd.concat(fr, ignore_index=True)
    for c in ["posteam", "defteam", "home_team"]:
        p[c] = p[c].replace(data.TEAM_MAP)
    return p


def team_unit_games(p):
    s = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.posteam.notna() & p.wp.between(0.05, 0.95)].copy()
    s["db"] = s.qb_dropback.fillna(0)
    s["deep"] = ((s.db == 1) & (s.air_yards >= 20)).astype(float)
    s["pass_epa"], s["rush_epa"], s["deep_epa"] = s.epa * s.db, s.epa * (1 - s.db), s.epa * s.deep
    g = s.groupby(["game_id", "season", "week", "posteam", "defteam"]).agg(
        plays=("epa", "size"), epa=("epa", "sum"), db=("db", "sum"), pass_epa=("pass_epa", "sum"),
        rush_epa=("rush_epa", "sum"), deep=("deep", "sum"), deep_epa=("deep_epa", "sum")).reset_index()
    return g


def _decay(df, key, cols, season, week, K):
    prev_max = df.loc[df.season == season - 1, "week"].max()
    m = ((df.season == season) & (df.week < week)) | (df.season == season - 1)
    d = df[m].assign(_w=_weights(df[m], season, week, prev_max))
    sums = {c: (d._w * d[c]).groupby(d[key]).sum() for c in cols}
    return sums, d


def player_logs(p):
    s = p[p.epa.notna() & p.wp.between(0.05, 0.95)]
    rec = s[(s.play_type == "pass") & s.receiver_player_id.notna()].groupby(["receiver_player_id", "season", "week"]).agg(
        tgt=("epa", "size"), tepa=("epa", "sum")).reset_index().rename(columns={"receiver_player_id": "pid"})
    rb = s[(s.play_type == "run") & s.rusher_player_id.notna() & (s.rusher_player_id != s.passer_player_id)].groupby(
        ["rusher_player_id", "season", "week"]).agg(car=("epa", "size"), cepa=("epa", "sum")).reset_index().rename(columns={"rusher_player_id": "pid"})
    def credits(cols_w, name):
        parts = []
        for c, w in cols_w:
            x = p[p[c].notna()][[c, "season", "week"]].rename(columns={c: "pid"})
            parts.append(x.assign(**{name: w}))
        return pd.concat(parts).groupby(["pid", "season", "week"])[name].sum().reset_index()
    sk = credits([("sack_player_id", 1.0), ("half_sack_1_player_id", 0.5), ("half_sack_2_player_id", 0.5)], "sacks")
    hit = credits([("qb_hit_1_player_id", 1.0), ("qb_hit_2_player_id", 1.0)], "hits")
    it = credits([("interception_player_id", 1.0)], "ints")
    pbu = credits([("pass_defense_1_player_id", 1.0), ("pass_defense_2_player_id", 1.0)], "pbu")
    logs = rec.merge(rb, on=["pid", "season", "week"], how="outer")
    for x in [sk, hit, it, pbu]:
        logs = logs.merge(x, on=["pid", "season", "week"], how="outer")
    logs = logs.fillna(0.0)
    logs["t"] = (logs.season * 100 + logs.week).astype("int64")
    logs = logs.sort_values(["pid", "t"])
    roll = logs.groupby("pid")[["tgt", "tepa", "car", "cepa", "sacks", "hits", "ints", "pbu"]].transform(
        lambda x: x.rolling(8, min_periods=1).sum())
    roll["games"] = logs.groupby("pid").cumcount().clip(upper=7) + 1
    logs = pd.concat([logs[["pid", "t"]], roll.add_prefix("r_")], axis=1)
    return logs, float(rec.tepa.sum() / rec.tgt.sum()), float(rb.cepa.sum() / rb.car.sum())


def availability(last, p):
    logs, lg_ept, lg_epc = player_logs(p)
    inj = pd.concat([pd.read_parquet(os.path.join(data.DATA_DIR, f"inj_{y}.parquet")) for y in range(FIRST, last + 1)], ignore_index=True)
    inj["team"] = inj.team.replace(data.TEAM_MAP)
    inj = inj[inj.report_status.isin(A.STATUS_W) & (inj.position != "QB") & inj.gsis_id.notna()].copy()
    inj["t"] = (inj.season * 100 + inj.week).astype("int64")
    m = pd.merge_asof(inj.sort_values("t"), logs.rename(columns={"pid": "gsis_id", "t": "t_l"}).sort_values("t_l"),
                      left_on="t", right_on="t_l", by="gsis_id", allow_exact_matches=False, direction="backward")
    m = m[(m.t - m.t_l) < 150].copy()
    g = m.r_games.clip(lower=1)
    ept = (m.r_tepa + 20 * lg_ept) / (m.r_tgt + 20)
    epc = (m.r_cepa + 30 * lg_epc) / (m.r_car + 30)
    sw = m.report_status.map(A.STATUS_W)
    m["rec"] = sw * (m.r_tgt / g) * (ept - (lg_ept - 0.15))
    m["rb"] = sw * (m.r_car / g) * (epc - (lg_epc - 0.05))
    m["pr"] = sw * 0.5 * (m.r_sacks / g * 1.7 + m.r_hits / g * 0.5)
    m["cov"] = sw * 0.5 * (m.r_ints / g * 4.0 + m.r_pbu / g * 0.8)
    out = m.groupby(["season", "week", "team"])[["rec", "rb", "pr", "cov"]].sum().reset_index()
    return out, m


def qb_blitz(last, p):
    yrs = [y for y in range(2022, last + 1) if os.path.exists(os.path.join(data.DATA_DIR, f"ftn_{y}.parquet"))]
    if not yrs:
        return None
    f = pd.concat([pd.read_parquet(os.path.join(data.DATA_DIR, f"ftn_{y}.parquet"), columns=["nflverse_game_id", "nflverse_play_id", "n_blitzers"]) for y in yrs])
    f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f["play_id"] = pd.to_numeric(f.play_id, errors="coerce")
    d = p[(p.qb_dropback == 1) & p.qb_epa.notna() & p.passer_player_id.notna() & (p.season >= 2022) & p.wp.between(0.05, 0.95)]
    d = d.merge(f, on=["game_id", "play_id"], how="inner")
    d["bl"] = (d.n_blitzers.fillna(0) > 0).astype(float)
    return d[["game_id", "season", "week", "posteam", "defteam", "passer_player_id", "qb_epa", "bl"]]


def build4(last):
    f3 = pd.read_parquet(f"{data.DATA_DIR}/features3.parquet")
    p = load(last)
    tg = team_unit_games(p)
    tgd = tg.rename(columns={"posteam": "o", "defteam": "posteam"})  # keyed by defense
    av, _ = availability(last, p)
    qb = qb_blitz(last, p)
    rows = []
    for (season, week), wk in f3.groupby(["season", "week"]):
        so, d_o = _decay(tg, "posteam", ["plays", "db", "deep", "deep_epa", "epa"], season, week, 4)
        sd, d_d = _decay(tgd, "posteam", ["plays", "deep", "deep_epa", "epa"], season, week, 4)
        pass_share = (so["db"] + 4 * 0.58 * 62) / (so["plays"] + 4 * 62)
        lg_deep = d_o.deep_epa.sum() / max(d_o.deep.sum(), 1)
        deep_rate = (so["deep"] + 4 * 0.1 * 36) / (so["db"] + 4 * 36)
        off_deep = (so["deep_epa"] + 8 * lg_deep) / (so["deep"] + 8) - lg_deep
        def_deep = (sd["deep_epa"] + 8 * lg_deep) / (sd["deep"] + 8) - lg_deep
        # recent form: last 4 games (raw EPA/play) minus decayed rating
        cur = tg[(tg.season == season) & (tg.week < week)]
        rec_o = cur.sort_values("week").groupby("posteam").tail(4).groupby("posteam").apply(lambda x: x.epa.sum() / x.plays.sum(), include_groups=False)
        rec_d = cur.sort_values("week").groupby("defteam").tail(4).groupby("defteam").apply(lambda x: x.epa.sum() / x.plays.sum(), include_groups=False)
        seas_o = so["epa"] / so["plays"]
        seas_d = sd["epa"] / sd["plays"]
        if qb is not None and season >= 2023:
            m = ((qb.season < season) | ((qb.season == season) & (qb.week < week)))
            q = qb[m]
            agg = q.groupby(["passer_player_id", "bl"]).qb_epa.agg(["sum", "size"]).unstack(fill_value=0)
            lgb = q[q.bl == 1].qb_epa.mean(); lgn = q[q.bl == 0].qb_epa.mean()
            qdelta = ((agg[("sum", 1.0)] + 60 * lgb) / (agg[("size", 1.0)] + 60)) - ((agg[("sum", 0.0)] + 60 * lgn) / (agg[("size", 0.0)] + 60)) - (lgb - lgn)
            dq = q[q.season >= season - 1]
            brate = (dq.groupby("defteam").bl.sum() + 20 * dq.bl.mean()) / (dq.groupby("defteam").size() + 20)
        else:
            qdelta, brate = None, None
        for g in wk.itertuples():
            h, a = g.home_team, g.away_team
            ps_h, ps_a = pass_share.get(h, .58), pass_share.get(a, .58)
            hO = ps_h * (g.h2_off_pass + g.a2_def_pass) + (1 - ps_h) * (g.h_off_rush + g.a_def_rush)
            aO = ps_a * (g.a2_off_pass + g.h2_def_pass) + (1 - ps_a) * (g.a_off_rush + g.h_def_rush)
            r = {"game_id": g.game_id, "unit_match": 62 * (hO - aO),
                 "deep_match": 36 * (deep_rate.get(h, .1) * (off_deep.get(h, 0) + def_deep.get(a, 0))
                                     - deep_rate.get(a, .1) * (off_deep.get(a, 0) + def_deep.get(h, 0))),
                 "off_form_diff": 62 * ((rec_o.get(h, np.nan) - seas_o.get(h, 0)) - (rec_o.get(a, np.nan) - seas_o.get(a, 0))),
                 "def_form_diff": -62 * ((rec_d.get(h, np.nan) - seas_d.get(h, 0)) - (rec_d.get(a, np.nan) - seas_d.get(a, 0)))}
            if qdelta is not None:
                hq = qdelta.get(g.home_qb_id, 0.0) if isinstance(g.home_qb_id, str) else 0.0
                aq = qdelta.get(g.away_qb_id, 0.0) if isinstance(g.away_qb_id, str) else 0.0
                r["qb_blitz_match"] = 36 * (brate.get(a, .25) * hq - brate.get(h, .25) * aq)
            rows.append(r)
        if week == 1:
            print("v4", season, flush=True)
    x = f3.merge(pd.DataFrame(rows), on="game_id", how="left")
    for side, team in [("h", "home_team"), ("a", "away_team")]:
        x = x.merge(av.rename(columns={"team": team, "rec": f"{side}_lost_rec", "rb": f"{side}_lost_rb", "pr": f"{side}_lost_pr", "cov": f"{side}_lost_cov"}),
                    on=["season", "week", team], how="left")
    for k in ["rec", "rb", "pr", "cov"]:
        x[f"avail_{k}_diff"] = -(x[f"h_lost_{k}"].fillna(0) - x[f"a_lost_{k}"].fillna(0))
    x["avail_total_diff"] = x[["avail_rec_diff", "avail_rb_diff", "avail_pr_diff", "avail_cov_diff"]].sum(axis=1)
    return x


if __name__ == "__main__":
    last = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    x = build4(last)
    x.to_parquet(f"{data.DATA_DIR}/features4.parquet")
    print(x.shape)
