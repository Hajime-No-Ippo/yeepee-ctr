#!/usr/bin/env python
"""v21trend/mktrend.py -- trend features (item smoothed trend, user trend). Leak-free: for each table with cut C only
train events/rows with ts<C (impressions+labels) and each target event's OWN click history (strictly before its ts). No eval aggregates.
Writes feat_trI/{table}.parquet (item), feat_trU/{table}.parquet (user + own-history), keyed by rowid."""
import sys, glob, time, numpy as np, polars as pl
H='/workspace/huawei'; J20=1737331200; DAY=86400
CACHE=f'{H}/v9d/val/cache_mx'
CUT={'f1':1,'f2':2,'f3':3,'f4':4,'f5':5,'f6':6,'val23':3,'val25':5,'val26':6,'eval':7}
def logit(x): return (x/(1-x)).log()
TR=pl.read_parquet(f'{H}/lite/train_rows.parquet'); TE=pl.read_parquet(f'{H}/lite/train_events.parquet')
ER=pl.read_parquet(f'{H}/lite/evaluation_rows.parquet'); EE=pl.read_parquet(f'{H}/lite/evaluation_events.parquet')
ROWS=TR.join(TE.select('eid','user','ts'),on='eid')            # all train rows w/ ts
EVC=TE.join(TR.group_by('eid').agg(pl.col('clicked').max().alias('ck')),on='eid')   # train events w/ any-click flag
g_all={}
def rowids(n):
    if n=='eval': return ER['rowid']
    g=[x for x in glob.glob(f'{CACHE}/{n}_*.parquet') if not x.endswith('.json')]; return pl.read_parquet(g[0],columns=['rowid'])['rowid']
def build(n):
    t0=time.time(); C=J20+CUT[n]*DAY; nd=CUT[n]
    rid=rowids(n).to_frame()
    if n=='eval': T=rid.join(ER,on='rowid').join(EE,on='eid')
    else: T=rid.join(TR.select('rowid','eid','item'),on='rowid').join(TE,on='eid')
    T=T.with_row_index('r')
    P=ROWS.filter(pl.col('ts')<C); g=float(P['clicked'].mean())
    # ---------------- item ----------------
    r1=pl.col('ts')>=C-DAY; r2=pl.col('ts')>=C-2*DAY
    it=P.group_by('item').agg(pl.len().alias('imp'),pl.col('clicked').sum().alias('clk'),
        r1.sum().alias('imp1'),pl.col('clicked').filter(r1).sum().alias('clk1'),r2.sum().alias('imp2'),pl.col('clicked').filter(r2).sum().alias('clk2'))
    ex=[]
    for k in (20,100):
        sl=(pl.col('clk')+k*g)/(pl.col('imp')+k); s1=(pl.col('clk1')+k*g)/(pl.col('imp1')+k); s2=(pl.col('clk2')+k*g)/(pl.col('imp2')+k)
        ex+=[(s1-sl).alias(f'trI_d1_k{k}'),(s2-sl).alias(f'trI_d2_k{k}'),(logit(s1)-logit(sl)).alias(f'trI_lo1_k{k}'),(logit(s2)-logit(sl)).alias(f'trI_lo2_k{k}')]
    sl=(pl.col('clk')+50*g)/(pl.col('imp')+50)
    ex+=[(((pl.col('clk1')+50*sl)/(pl.col('imp1')+50))-sl).alias('trI_eb1'),(((pl.col('clk2')+50*sl)/(pl.col('imp2')+50))-sl).alias('trI_eb2'),
         (pl.col('imp1')/pl.col('imp')).alias('trI_share1'),(pl.col('imp2')/pl.col('imp')).alias('trI_share2'),
         (((pl.col('clk')+20*g)/(pl.col('imp')+20))).alias('_sl20'),pl.col('imp').alias('_imp')]
    it=it.with_columns(ex).select(['item']+[c for c in it.with_columns(ex).columns if c.startswith(('trI','_'))])
    R=T.select('r','rowid','eid','item','user','ts').join(it,on='item',how='left').sort('r')
    icols=[c for c in R.columns if c.startswith('trI')]
    ev=[]
    for c in ('trI_d1_k100','trI_eb1','trI_lo1_k20'):
        x=pl.col(c); ev+=[x.rank('average').over('eid').alias(f'{c}_rk'),((x-x.mean().over('eid'))/(x.std().over('eid')+1e-9)).alias(f'{c}_z')]
    R=R.with_columns(ev); icols+=[f'{c}_{s}' for c in ('trI_d1_k100','trI_eb1','trI_lo1_k20') for s in ('rk','z')]
    R=R.with_columns(pl.lit(nd).alias('trI_nd')); icols.append('trI_nd')
    R.select(['rowid']+icols).with_columns([pl.col(c).cast(pl.Float32) for c in icols]).write_parquet(f'{H}/v21trend/feat_trI/{n}.parquet')
    # ---------------- user (train events before C) ----------------
    PE=EVC.filter(pl.col('ts')<C); ge=float(PE['ck'].mean())
    us=PE.group_by('user').agg(pl.len().alias('nev'),pl.col('ck').sum().alias('nck'),r1.sum().alias('nev1'),pl.col('ck').filter(r1).sum().alias('nck1'),
        r2.sum().alias('nev2'),pl.col('ck').filter(r2).sum().alias('nck2'))
    ex=[]
    for k in (2,8):
        sl=(pl.col('nck')+k*ge)/(pl.col('nev')+k); s1=(pl.col('nck1')+k*ge)/(pl.col('nev1')+k); s2=(pl.col('nck2')+k*ge)/(pl.col('nev2')+k)
        ex+=[(s1-sl).alias(f'trU_d1_k{k}'),(s2-sl).alias(f'trU_d2_k{k}'),(logit(s1)-logit(sl)).alias(f'trU_lo1_k{k}'),(logit(s2)-logit(sl)).alias(f'trU_lo2_k{k}')]
    ex+=[(pl.col('nev1')/(pl.col('nev')/nd)).alias('trU_act1'),(pl.col('nev2')/(2*pl.col('nev')/nd)).alias('trU_act2'),pl.col('nev1').alias('trU_nev1'),pl.col('nev2').alias('trU_nev2'),
         (pl.col('nck1')/(pl.col('nck')/nd+1e-3)).alias('trU_ckact1')]
    us=us.with_columns(ex); ucols=[c for c in us.columns if c.startswith('trU')]
    # ---------------- own history (per target event) ----------------
    E=T.select('eid','user','ts','h_items','h_ts').unique('eid')
    L=(E.select('eid','ts',pl.col('h_items').alias('hi'),pl.col('h_ts').alias('ht')).explode(['hi','ht']).drop_nulls('hi')
        .with_columns(pl.col('ht').cast(pl.Int64)).filter(pl.col('ht')<pl.col('ts'))
        .join(it.select('item',pl.col('_sl20'),pl.col('_imp')),left_on='hi',right_on='item',how='left')
        .sort('eid','ht').with_columns(pl.col('ht').cum_count().over('eid').alias('_o'),pl.len().over('eid').alias('_n')))
    L=L.with_columns((pl.col('_n')-pl.col('_o')).alias('pos'), pl.col('_imp').fill_null(0).log1p().alias('lp'))   # pos 1 = newest (cum_count is 1-based)
    L=L.with_columns((pl.col('_o')-1).alias('_oo'))
    nw=pl.col('pos')<=5
    ha=L.group_by('eid').agg(pl.len().alias('n'),pl.col('ts').first().alias('ts'),pl.col('ht').min().alias('h0'),pl.col('ht').max().alias('hN'),
        (pl.col('ht')>=pl.col('ts')-DAY).sum().alias('n1'),(pl.col('ht')>=pl.col('ts')-3*DAY).sum().alias('n3'),
        pl.col('ht').filter(pl.col('pos')==2).first().alias('_h2'),   # second newest
        pl.col('lp').filter(nw).mean().alias('lp_new'),pl.col('lp').filter(~nw).mean().alias('lp_old'),
        pl.col('_sl20').filter(nw).mean().alias('sl_new'),pl.col('_sl20').filter(~nw).mean().alias('sl_old'),
        pl.col('ht').sort().median().alias('hmid'))
    span=((pl.col('ts')-pl.col('h0'))/DAY).clip(0.25,None)
    ha=ha.with_columns(span.alias('_span')).with_columns(
        pl.col('n1').alias('trH_n1'),pl.col('n3').alias('trH_n3'),pl.col('_span').alias('trH_span'),
        (pl.col('n')/pl.col('_span')).alias('trH_rate'),
        (pl.col('n1')/(pl.col('n')/pl.col('_span'))).alias('trH_trend1'),
        ((pl.col('n3')/3)/(pl.col('n')/pl.col('_span'))).alias('trH_trend3'),
        (((pl.col('ts')-pl.col('hN'))/DAY)/(pl.col('_span')/pl.col('n'))).alias('trH_gapratio'),
        (((pl.col('hN')-pl.col('_h2'))/DAY)/(pl.col('_span')/pl.col('n'))).alias('trH_lastgapratio'),
        (pl.col('lp_new')-pl.col('lp_old')).alias('trH_pop_d'),
        (pl.col('sl_new')-pl.col('sl_old')).alias('trH_ctr_d'),pl.col('lp_new').alias('trH_pop_new'))
    # half-split click rate ratio: newer half vs older half
    L2=L.join(ha.select('eid','hmid'),on='eid').with_columns((pl.col('ht')>pl.col('hmid')).alias('_new'))
    hh=L2.group_by('eid').agg((pl.col('_new').sum()).alias('nn'),(~pl.col('_new')).sum().alias('no'),pl.col('ht').filter(pl.col('_new')).min().alias('_tn'),pl.col('ht').filter(~pl.col('_new')).min().alias('_to'))
    hh=hh.join(ha.select('eid','ts','hmid'),on='eid').with_columns(
        (((pl.col('nn'))/((pl.col('ts')-pl.col('hmid'))/DAY).clip(0.1,None))/((pl.col('no')+1e-9)/((pl.col('hmid')-pl.col('_to'))/DAY).clip(0.1,None))).log().alias('trH_halfratio'))
    ha=ha.join(hh.select('eid','trH_halfratio'),on='eid',how='left')
    hcols=['trH_n1','trH_n3','trH_span','trH_rate','trH_trend1','trH_trend3','trH_gapratio','trH_lastgapratio','trH_pop_d','trH_ctr_d','trH_pop_new','trH_halfratio']
    U=T.select('r','rowid','eid','user').join(us.select(['user']+ucols),on='user',how='left').join(ha.select(['eid']+hcols),on='eid',how='left').sort('r')
    U=U.with_columns([pl.when(pl.col(c).cast(pl.Float64).is_infinite()).then(None).otherwise(pl.col(c).cast(pl.Float64)).alias(c) for c in hcols+ucols])
    allc=ucols+hcols
    U.select(['rowid']+allc).with_columns([pl.col(c).cast(pl.Float32) for c in allc]).write_parquet(f'{H}/v21trend/feat_trU/{n}.parquet')
    print(n,'C',C,'rows',T.height,'item feats',len(icols),'user feats',len(allc),f'{time.time()-t0:.0f}s',flush=True)
if __name__=='__main__':
    for n in (sys.argv[1:] or list(CUT)): build(n)
