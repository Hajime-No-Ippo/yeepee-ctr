exec(open('/tmp/dig/lat.py').read())
print('row length stats: all clicks per user', ALL.group_by('u').len()['len'].describe().filter(pl.col('statistic').is_in(['mean','50%','max'])).rows())
win=ALL.filter(pl.col('t')>=J20); j13=ALL.filter(pl.col('t')>=J20-7*DAY); l30=ALL.filter(pl.col('L')-pl.col('p')<=30)
print('window clicks/user quantiles',win.group_by('u').len()['len'].quantile(.5),win.group_by('u').len()['len'].quantile(.99),win.group_by('u').len()['len'].max())
for nm,df in (('l30',l30),('win',win),('j13',j13),('all',ALL)):
    for beta in (0,0.5,1.0):
        s,vt=svdv(mat(df,0.5,beta),16)
        x=cosf(vt[1:13].T*s[1:13]); 
        dd=D.select('eid','clicked','cut').with_columns(pl.Series('x',x))
        print(nm,'beta',beta,'sv',np.round(s[:16]/s[15],2),'within',{c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)},flush=True)
