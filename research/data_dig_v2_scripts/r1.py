exec(open('/tmp/dig/lat.py').read())
Vt=np.load('/tmp/dig/val_t.npz')
x_svd=cosf(np.load('V_ab55.npy')); np.save('x_svd_ab55.npy',x_svd)
ev(x_svd,'svd a.5 b.5 all-clicks k13 cos')
ev(np.load('x_l40.npy') if False else cosf(np.load('V13.npy')),'v1 V13 (last30) cos')
for tag in ('nbnn','nb','nbu'):
    Z=np.load(f'tt_{tag}.npz'); ev(Z['dot'],f'twotower {tag} dot')
Z=np.load('tt_nbnn.npz'); s=Z['dot']
G=Vt['evix']*2+(cut==26); gu,gk=np.unique(G,return_inverse=True)
df=pl.DataFrame({'g':gk,'s':s}).with_columns(pl.col('s').exp().sum().over('g').log().alias('lse'),pl.col('s').mean().over('g').alias('mn'),pl.col('s').max().over('g').alias('mx'),
   (pl.col('s')-pl.col('s').mean().over('g')).alias('dm'))
np.save('x_tt.npy',np.c_[s,df['lse'].to_numpy(),df['mn'].to_numpy(),df['dm'].to_numpy()])
ev(df['lse'].to_numpy(),'event lse broadcast')
print('stack tt dot+lse+mean+dmean', resid(np.load('x_tt.npy'),3))
print('stack tt dot + svd cos', resid(np.c_[s,x_svd],3))
print('stack tt 4 + svd + v1 V13', resid(np.c_[np.load('x_tt.npy'),x_svd,cosf(np.load('V13.npy'))],3))
