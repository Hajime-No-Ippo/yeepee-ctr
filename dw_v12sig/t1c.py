import numpy as np, polars as pl
S=pl.read_parquet('cache/sig_c27.parquet'); r=pl.read_parquet('t1_resid.parquet')
H='/workspace/huawei'; ev=pl.read_parquet(f'{H}/v10/feat_v3/eval.parquet',columns=['rowid','ii'])
# sigs order = extend_eval order (sorted by ts,eid) -> join on rowid
d=S.join(r,on='rowid').join(ev,on='rowid'); print(d.height)
for c in ['cl_all_lg','cl_all_z','cl_all_dm','cl_all_s20_dm','cl_l48_dm','lc_agecurve','lc_logage','lc_rel24','sq_l1_k1_pmi_z','rp_item_rate']:
    x=d[c].to_numpy(); m=np.isfinite(x); print('%-16s row-level corr(res_w)=%+.3f  corr(res)=%+.3f  corr(res_ev)=%+.3f'%(c,np.corrcoef(x[m],d['res_w'].to_numpy()[m])[0,1],np.corrcoef(x[m],d['res'].to_numpy()[m])[0,1],np.corrcoef(x[m],d['res_ev'].to_numpy()[m])[0,1]))
for c in ['cl_all_logD','cl_all_mx','cl_l48_logD']:
    x=d[c].to_numpy(); print(c,'corr res_ev %+.3f'%np.corrcoef(x,d['res_ev'].to_numpy())[0,1])
# item-level
g=d.group_by('ii').agg(pl.col('res_w').mean().alias('w'),pl.len().alias('n'),pl.col('cl_all_dm').mean().alias('cl'),pl.col('cl_all_lg').first().alias('lg')).filter(pl.col('n')>=8)
print('item-level corr(mean res_w, cl logg)',np.corrcoef(g['w'],g['lg'])[0,1],g.height)
