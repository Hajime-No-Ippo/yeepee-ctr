exec(open('/tmp/dig/lat.py').read())
for a in (0.3,0.5,0.7):
  for beta in (0.25,0.5,0.75):
    s,vt=svdv(mat(ALL,a,beta),16)
    for k in (12,13):
        x=cosf(vt[1:k].T*s[1:k]); dd=D.select('eid','clicked','cut').with_columns(pl.Series('x',x))
        print('a',a,'beta',beta,'k',k,{c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)},flush=True)
    if a==0.5 and beta==0.5: np.save('V_ab55.npy',vt[1:13].T*s[1:13]); np.save('Vfull_ab55.npy',vt.T*s)
V=nrm(np.load('V_ab55.npy'))
rng=np.random.default_rng(0)
A=ALL.with_columns((pl.col('t')>=J20).alias('w'))
cnt=A.group_by('u').agg(pl.col('w').sum().alias('nw'),(~pl.col('w')).sum().alias('nn'))
us=cnt.filter((pl.col('nw')>=8)&(pl.col('nn')>=8))['u'].to_numpy()
A=A.filter(pl.col('u').is_in(us)); A=A.with_columns(pl.Series('r',rng.random(A.height)))
def cent(df):
    g=df.group_by('u').agg('ix').sort('u'); return g['u'].to_numpy(),np.stack([nrm(V[np.array(l)].mean(0,keepdims=True))[0] for l in g['ix'].to_list()])
u1,c1=cent(A.filter(~pl.col('w')));u2,c2=cent(A.filter(pl.col('w')))
u3,c3=cent(A.filter(~pl.col('w')&(pl.col('r')<.5)));u4,c4=cent(A.filter(~pl.col('w')&(pl.col('r')>=.5)))
A2=A.filter(~pl.col('w')).with_columns((pl.col('t')>=pl.col('t').median().over('u')).alias('late'))
u5,c5=cent(A2.filter(~pl.col('late')));u6,c6=cent(A2.filter(pl.col('late')))
print('users',len(us))
print('cos warm vs window (same user)',np.round(np.median((c1*c2).sum(1)),3),'random-user',np.round(np.median((c1*c2[rng.permutation(len(c2))]).sum(1)),3))
m=np.intersect1d(u3,u4); print('cos warm random halves',np.round(np.median((c3[np.isin(u3,m)]*c4[np.isin(u4,m)]).sum(1)),3))
m=np.intersect1d(u5,u6); print('cos warm early-half vs late-half',np.round(np.median((c5[np.isin(u5,m)]*c6[np.isin(u6,m)]).sum(1)),3))
