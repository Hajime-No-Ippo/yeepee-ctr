#!/usr/bin/env python
"""v21trend/mkdecay.py -- Phase 2c: time-decayed target encoding (feat_dec). Leak-free: only train rows/events with ts<C (labels from the fit window), decay weight exp(-(C-ts)ln2/hl),
hl in {6h,12h,24h}; smoothed toward base rate g (item k=20,100; user k=50,200 impressions; user x item k=20). Added alongside existing 1-3 day features (not replacing).
Shares the data loading / target-frame code of mktrend.py. Slate versions: rank, z, gap-to-event-mean."""
import sys, time, numpy as np, polars as pl
from mktrend import ROWS, EVC, TE, TR, ER, EE, rowids, J20, DAY, CUT, logit
H='/workspace/huawei'; LN2=float(np.log(2)); HLS={'6':6*3600,'12':12*3600,'24':24*3600}
def build(n):
    t0=time.time(); C=J20+CUT[n]*DAY; rid=rowids(n).to_frame()
    if n=='eval': T=rid.join(ER,on='rowid').join(EE,on='eid')
    else: T=rid.join(TR.select('rowid','eid','item'),on='rowid').join(TE,on='eid')
    T=T.with_row_index('r').select('r','rowid','eid','item','user')
    P=ROWS.filter(pl.col('ts')<C); g=float(P['clicked'].mean())
    P=P.with_columns([(-(C-pl.col('ts'))*LN2/hl).exp().alias(f'w{h}') for h,hl in HLS.items()])
    P=P.with_columns([(pl.col(f'w{h}')*pl.col('clicked')).alias(f'wc{h}') for h in HLS])
    aggs=[pl.col(f'w{h}').sum().alias(f'di{h}') for h in HLS]+[pl.col(f'wc{h}').sum().alias(f'dc{h}') for h in HLS]
    # long smoothed ctr for the diff features
    it=P.group_by('item').agg(aggs+[pl.len().alias('imp'),pl.col('clicked').sum().alias('clk')])
    ex=[]
    for h in HLS:
        for k in (20,100): ex.append(((pl.col(f'dc{h}')+k*g)/(pl.col(f'di{h}')+k)).alias(f'dec_it{h}_ctr_k{k}'))
        ex.append(pl.col(f'di{h}').alias(f'dec_it{h}_imp'))
        ex.append((((pl.col(f'dc{h}')+20*g)/(pl.col(f'di{h}')+20))-((pl.col('clk')+20*g)/(pl.col('imp')+20))).alias(f'dec_it{h}_dlong'))
    it=it.with_columns(ex).select(['item']+[c for c in it.with_columns(ex).columns if c.startswith('dec_')])
    R=T.join(it,on='item',how='left')
    # user level (impression-based)
    us=P.group_by('user').agg(aggs); ex=[]
    for h in HLS:
        for k in (50,200): ex.append(((pl.col(f'dc{h}')+k*g)/(pl.col(f'di{h}')+k)).alias(f'dec_us{h}_ctr_k{k}'))
        ex.append((pl.col(f'di{h}')/10).alias(f'dec_us{h}_nev')); ex.append(pl.col(f'dc{h}').alias(f'dec_us{h}_clk'))
    us=us.with_columns(ex); us=us.select(['user']+[c for c in us.columns if c.startswith('dec_')])
    R=R.join(us,on='user',how='left')
    # user x item
    ui=P.group_by('user','item').agg(aggs); ex=[]
    for h in HLS:
        ex+= [((pl.col(f'dc{h}')+20*g)/(pl.col(f'di{h}')+20)).alias(f'dec_ui{h}_ctr'),pl.col(f'di{h}').alias(f'dec_ui{h}_imp'),pl.col(f'dc{h}').alias(f'dec_ui{h}_clk')]
    ui=ui.with_columns(ex); ui=ui.select(['user','item']+[c for c in ui.columns if c.startswith('dec_')])
    R=R.join(ui,on=['user','item'],how='left').sort('r')
    # nulls -> 0 for counts, NaN stays for ctr (tree handles) except ui: pair never shown
    cnt=[c for c in R.columns if c.endswith(('_imp','_clk','_nev'))]; R=R.with_columns([pl.col(c).fill_null(0) for c in cnt])
    main=['dec_it6_ctr_k20','dec_it12_ctr_k20','dec_it6_dlong','dec_it12_dlong','dec_ui12_ctr','dec_ui6_imp']
    ev=[]
    for c in main:
        x=pl.col(c).cast(pl.Float64); ev+=[x.rank('average').over('eid').alias(c+'_rk'),((x-x.mean().over('eid'))/(x.std().over('eid')+1e-9)).alias(c+'_z'),(x-x.mean().over('eid')).alias(c+'_gap')]
    R=R.with_columns(ev); cols=[c for c in R.columns if c.startswith('dec_')]
    R.select(['rowid']+cols).with_columns([pl.col(c).cast(pl.Float32) for c in cols]).write_parquet(f'{H}/v21trend/feat_dec/{n}.parquet')
    print(n,'C',C,'feats',len(cols),f'{time.time()-t0:.0f}s',flush=True)
if __name__=='__main__':
    import os; os.makedirs(f'{H}/v21trend/feat_dec',exist_ok=True)
    for n in (sys.argv[1:] or list(CUT)): build(n)
