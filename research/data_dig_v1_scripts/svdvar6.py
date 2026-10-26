exec(open('svdvar5.py').read().split('variants={')[0])
A=ALL.sort('u','t').with_columns(pl.int_range(pl.len()).over('u').alias('p'),pl.len().over('u').alias('L'))
for N in (10,20,30,40,80):
    dfv=A.filter(pl.col('L')-pl.col('p')<=N).unique(['u','ix'])
    s,vt=build(dfv['u'].to_numpy(),dfv['ix'].to_numpy(),LH.height,0.5,24)
    out=[]
    for kk in (12,13,16):
        x,_=cosf(vt[1:kk].T*s[1:kk]); dd=D.with_columns(pl.Series('x',x))
        out.append((kk,{c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}))
    print('lastN',N,'pairs',dfv.height,'top sv',np.round(s[:15],2),out,flush=True)
