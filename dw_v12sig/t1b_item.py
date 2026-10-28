import numpy as np, polars as pl
H='/workspace/huawei'
ev=pl.read_parquet(f'{H}/v10/feat_v3/eval.parquet'); r=pl.read_parquet('t1_resid.parquet')
d=ev.join(r,on='rowid'); 
g=d.group_by('ii').agg(pl.col('res_w').mean().alias('w'),pl.col('res').mean().alias('r'),pl.len().alias('n'),pl.col('za').mean().alias('za'),pl.col('zb').mean().alias('zb'),
 *[pl.col(c).first().alias(c) for c in ['it_imp','it_clk','it_ctr','it_ctr24','it_ctr6','it_trend24','it_imp24','it_imp_rate','it_nclk_all','it_nusers','it_nclk_warm','it_clk_dec2d','it_imptrend24']])
g=g.filter(pl.col('n')>=8); print(g.height)
for c in g.columns[4:]: 
    x=g[c].to_numpy().astype(float); m=np.isfinite(x)
    print('%-16s corr(w)=%+.3f corr(r)=%+.3f  corr(zb-za)=%+.3f'%(c,np.corrcoef(x[m],g['w'].to_numpy()[m])[0,1],np.corrcoef(x[m],g['r'].to_numpy()[m])[0,1],np.corrcoef(x[m],(g['zb']-g['za']).to_numpy()[m])[0,1]))
print('corr(n_eval_rows, w)',np.corrcoef(g['n'],g['w'])[0,1], 'corr log n',np.corrcoef(np.log(g['n']),g['w'])[0,1])
print('sd of item mean w',g['w'].std(),' expected noise sd',(d['res_w'].std()/np.sqrt(g['n'])).mean())
