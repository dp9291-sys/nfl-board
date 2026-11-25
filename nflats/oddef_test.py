"""Walk-forward tests of offense/defense EPA signals vs closing spreads, 2015-26, BH q=0.10 (Nov 25, 2026). Run: PYTHONPATH=. python -m nflats.oddef_test"""
import pickle, numpy as np, pandas as pd, json
from nflats import data, validate as V
d,_,_=pickle.load(open(f"{data.DATA_DIR}/v3_signals.pkl","rb"))
d=d.copy(); d["ats"]=(d.result-d.spread_line).clip(-21,21); d["ats_raw"]=d.result-d.spread_line
def net(k): return (d[f"h2_off_{k}"]-d[f"h2_def_{k}"])-(d[f"a2_off_{k}"]-d[f"a2_def_{k}"])
def netg(k): return (d[f"h2_off_{k}"]-d[f"h2_def_{k}"])-(d[f"a2_off_{k}"]-d[f"a2_def_{k}"])
for k in ["epa","sr","early","nto","pass","expl"]: d[f"n_{k}"]=net(k)
d["n_sack"]=(d.h2_def_sack-d.h2_off_sack)-(d.a2_def_sack-d.a2_off_sack)
d["off_diff"]=d.h2_off_epa-d.a2_off_epa; d["def_diff"]=d.a2_def_epa-d.h2_def_epa
d["epa_minus_sr"]=d.n_epa-(d.n_epa.std()/d.n_sr.std())*d.n_sr
C={"Net EPA/play":["n_epa"],"Net success rate":["n_sr"],"Early-down net EPA":["n_early"],"Turnover-free net EPA":["n_nto"],
   "Pass EPA net":["n_pass"],"Explosive-play rate net":["n_expl"],"Sack rate net":["n_sack"],
   "Offense only (EPA)":["off_diff"],"Defense only (EPA)":["def_diff"],"Offense and defense weighted separately":["off_diff","def_diff"],
   "EPA beyond success rate (big-play luck)":["epa_minus_sr"]}
T=list(range(2015,2027)); m=d.season.isin(T); rows=[]
for name,cols in C.items():
    d[cols]=d[cols].fillna(0)
    a=V.oos_adjust(d,cols,T); s=V.score(d[m],a[m])
    b=m&(a.abs()>=0.5)&(d.ats_raw!=0); w=(np.sign(a)==np.sign(d.ats_raw))[b]
    b26=b&(d.season==2026)
    rows.append(dict(signal=name,t=s["t"],p=round(s["p"],3),top_third=s["top_hit"],lean05_n=int(b.sum()),lean05_win=round(100*w.mean(),1) if b.any() else None,
                     n26=int(b26.sum()),win26=round(100*(np.sign(a)==np.sign(d.ats_raw))[b26].mean(),1) if b26.any() else None))
t=pd.DataFrame(rows); t["fdr"]=V.bh(t.p.values,0.10)
pd.set_option('display.width',220); print(t.sort_values("t",ascending=False).to_string(index=False))
t.to_json("/home/claude/data/oddef_results.json",orient="records")
