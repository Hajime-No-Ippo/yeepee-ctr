exec(open('svdvar4.py').read().split('s,vt=build(full')[0].replace("print('clicks per user","pass #"))
LH=EV.sort('ts').group_by('user').agg(pl.col('h_items').last(),pl.col('h_ts').last()).collect().with_row_index('u')
X=LH.explode('h_items','h_ts').rename({'h_items':'item'}).join(imap,on='item').with_columns(pl.int_range(pl.len()).over('u').alias('p'),pl.len().over('u').alias('L'))
ALL=EV.select('user','h_items','h_ts').explode('h_items','h_ts').unique(['user','h_items','h_ts']).collect().rename({'h_items':'item','h_ts':'t'}).join(imap,on='item').join(LH.select('user','u'),on='user')
variants={
 'last10': X.filter(pl.col('L')-pl.col('p')<=10),
 'last20_dup': X,
 'win_J13_C': ALL.filter(pl.col('t')>=J20-7*86400),
 'win_J17_C': ALL.filter(pl.col('t')>=J20-3*86400),
 'warm_only_lt_J20': ALL.filter(pl.col('t')<J20),
 'win_J20_C': ALL.filter(pl.col('t')>=J20),
}
for n,dfv in variants.items():
    dfv=dfv.unique(['u','ix'])
    s,vt=build(dfv['u'].to_numpy(),dfv['ix'].to_numpy(),LH.height,0.5,24)
    print(n,'pairs',dfv.height,'spectrum',np.round(s[:18],2))
    for kk in (12,13,16,20):
        x,_=cosf(vt[1:kk].T*s[1:kk]); dd=D.with_columns(pl.Series('x',x))
        wa={c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}
        print(f'   k{kk} within',wa,flush=True)
