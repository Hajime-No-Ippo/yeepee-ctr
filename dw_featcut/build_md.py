import sys; sys.path.insert(0,'.')
from common import *
import pandas as pd, os
N=pd.read_csv('gain_table.csv',index_col=0)
chk=json.load(open('check_results.json')) if os.path.exists('check_results.json') else []
INFO={ # family -> (source dir, builder, description)
'feat_tt':('v11/feat_tt','v17run/src/ttfeat.py','two-tower/learned topic-embedding scores of user history vs candidate (tA/tB dot, cos, lse, mean, within-event dmean/dmax/z)'),
'feat_Ap':('v17cooc/feat_Ap (v17cooc/mkcooc.py)','v17run/src/cooc/mkcooc.py','co-occurrence PPMI-embedding cos of history vs candidate (pp8/12/16 variants, ppb, x-variant), unit-norm, ucos_dot, 12 cos-products'),
'feat_sqC':('v11/feat_sqC','v17run/src/seq11.py, avg_sq.py','sequence-model (transformer) averaged logit/gap/attention stats; raw pdot/pcos/mdot/mcos pre-dropped by --drop'),
'feat_dw':('v12/feat_dw','v17run/src/dw/mkdw.py','DW features: impression/skip SVD cosines, joint cos, history-cos in impression space, click/impression centroids + slate coherence, c_* contrast features'),
'feat_lr2':('v11/feat_lr2','v17run/src/lrfeat2.py','low-rank (SVD k=4..16) user-history vs item cosines, contrast variants (con05/con1), within-event dmax'),
'feat_NAs':('v19/feat_NAs','v19/mkmf.py','NMF(12) user-item and item-item-PPMI affinities/lifts/entropy/cos per-dim (variant A, train-only graph)'),
'feat_x':('v11/feat_x','v17run/src/extra.py','extra click-graph features: con/conr/crcos/icon/icos at k=32/64/256, cluster CTR + lift + imp counts'),
'feat_pp':('v11/feat_pp','v17run/src/ppmi.py','PPMI-embedding history cos / last3 / con05 features (+z, +rk)'),
'v2_':('base cache v8-lite (v9d/val/cache_mx)','v17run/src/build_folds.py (v2 dir)','v2 behavioural stats: item imp/clk/ctr windows, ip (item-pair) counts+decays, user stats, user-item since/noclk, history len/gaps, co_* sums'),
'ev_*':('feat_v3 / base cache','v17run/src/feats.py, v3feat_ne.py','within-event transforms (rk, z, max, mean) of hist/co/sv/it/mf groups'),
'sv_*':('feat_v3 / base cache','v17run/src/feats.py','SVD (k=#) user-history / item cos, dot, norm, max, con'),
'it_*':('feat_v3 / base cache','v17run/src/feats.py','item impression/click counts, CTR, trends, decay'),
'us_*':('feat_v3 / base cache','v17run/src/feats.py','user event counts, CTR, history length, cold flags, recency'),
'hist_*':('feat_v3 / base cache','v17run/src/feats.py','user-item history counts/positions (n_all, noclk, since_*, in)'),
'co_*':('feat_v3 / base cache','v17run/src/v3feat_ne.py','user-history co-click cos / transition stats (max5, mean5, last, rw, sum5)'),
'mf_*':('feat_v3 / base cache','v17run/src/feats.py','matrix-factorisation score/dot/bias'),
}
ORDER=['feat_tt','feat_Ap','feat_sqC','feat_dw','feat_lr2','feat_NAs','feat_x','feat_pp','v2_','ev_*','sv_*','it_*','us_*','hist_*','co_*','mf_*']
rows=[]
for f in ORDER:
    g=N[N.family==f]; s,b,d=INFO[f]
    rows.append(f"| {f} | {len(g)} | {g.mean_all.sum()*100:.1f}% | {(g.tier=='safe').sum()} | {(g.tier=='probably_safe').sum()} | {len(g)-(g.tier!='keep').sum()} | `{s}` | `{b}` | {d} |")
inv='\n'.join(rows)
tot=f"| **total** | {len(N)} | 100% | {(N.tier=='safe').sum()} | {(N.tier=='probably_safe').sum()} | {(N.tier=='keep').sum()} | | | |"
def tbl(t):
    g=N[N.tier==t].sort_values(['family','mean_all'])
    out=['| feature | family | max gain share (5 runs) | rank | evidence |','|---|---|---|---|---|']
    for c,r in g.iterrows():
        ev=[]
        if r.max_all<2e-5: ev.append('~zero gain in all 5 runs')
        elif r.max_all<1e-4: ev.append('gain <1e-4 in all 5 runs')
        if isinstance(r.redundant_with,str): ev.append(f'|Spearman|={r.redundant_rho:.3f} with `{r.redundant_with}` (kept: higher gain)')
        if r.red_both: ev.append('also dropped by v20 redundancy.json')
        out.append(f"| `{c}` | {r.family} | {r.max_all:.1e} | {int(r.rank_A)} | {'; '.join(ev)} |")
    return '\n'.join(out)
top=N.sort_values('mean_all',ascending=False).head(15)
toptxt='\n'.join(f"- `{c}` ({r.family}) {r.mean_all*100:.1f}%" for c,r in top.iterrows())
ck=''
if chk:
    ck='\n'.join(f"| {r['set']} | {r['nf']} | {r['seed']} | {r['auc'].get('150')} | {r['auc'].get('300')} | {r['fit_s']} |" for r in chk)
open('_parts.json','w').write(json.dumps(dict(inv=inv,tot=tot,safe=tbl('safe'),prob=tbl('probably_safe'),top=toptxt,ck=ck)))
