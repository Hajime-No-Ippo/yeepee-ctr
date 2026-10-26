exec(open('svdvar5.py').read().split('variants={')[0])
dfv=X.unique(['u','ix']); s,vt=build(dfv['u'].to_numpy(),dfv['ix'].to_numpy(),LH.height,0.5,13)
V=vt[1:13].T*s[1:13]; V/=np.linalg.norm(V,axis=1,keepdims=True)+1e-9; np.save('V13.npy',V)
A=ALL.with_columns((pl.col('t')<J20).alias('warm'))
rng=np.random.default_rng(0); us=rng.choice(A['u'].unique().to_numpy(),3000,replace=False)
A=A.filter(pl.col('u').is_in(us))
res={'ww':[],'nn':[],'wn':[],'rand':[]}
for u,g in A.group_by('u'):
    vi=V[g['ix'].to_numpy()]; w=g['warm'].to_numpy()
    S=vi@vi.T; iu=np.triu_indices(len(vi),1)
    a=w[iu[0]]; b=w[iu[1]]
    res['ww']+=list(S[iu][a&b]); res['nn']+=list(S[iu][~a&~b]); res['wn']+=list(S[iu][a!=b])
    r=V[rng.integers(0,len(V),len(vi))]; res['rand'].append((vi@r.T).mean())
print({k:(round(float(np.mean(v)),4),len(v)) for k,v in res.items()})
