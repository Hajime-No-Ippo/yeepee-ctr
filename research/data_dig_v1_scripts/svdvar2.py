exec(open('svdvar.py').read().split('res={}')[0])
for f in ['x_con05_k32','x_con05_k32_z','v2_svd16_cos','sv_k16_udot','sv_k16_ucos','sv_k64_con05','x_icon05_k64']:
    print(f,{c:round(wauc(D.filter(pl.col('cut')==c),f),4) for c in (25,26)}, {c:round(roc_auc_score(y[cut==c],D.filter(pl.col('cut')==c)[f].fill_null(-9).to_numpy()),4) for c in (25,26)})
for cfg in [(4,0.5,1,0),(8,0.5,1,0),(8,1.0,1,0),(12,0.5,1,0),(12,1.0,1,0),(16,1.0,1,0),(20,0.5,1,0),(24,0.5,1,0)]:
    x=feat(*cfg); dd=D.with_columns(pl.Series('x',x))
    sa={c:round(roc_auc_score(y[cut==c],x[cut==c]),4) for c in (25,26)}
    wa={c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}
    r=resid(x); print('k,alpha,binary,recent',cfg,'glob',sa,'within',wa,'resid delta',r,flush=True)
    np.save(f'x_{"_".join(map(str,cfg))}.npy',x)
