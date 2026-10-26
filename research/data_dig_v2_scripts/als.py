# 1(b) implicit weighted ALS (Hu-Koren-Volinsky) at d=8/12/16 on all pre-J25 clicks; 1(d) per-user ridge on fixed item vectors
exec(open('/tmp/dig/lat.py').read())
df=ALL.unique(['u','ix']); R=sp.csr_matrix((np.ones(df.height,np.float32),(df['u'].to_numpy(),df['ix'].to_numpy())),shape=(NU,NI))
def side(Rm,Y,alpha,lam):
    d=Y.shape[1]; Rc=Rm.tocoo(); YtY=Y.T@Y
    O=(Y[Rc.col][:,:,None]*Y[Rc.col][:,None,:]).reshape(len(Rc.col),d*d)
    S=sp.csr_matrix((np.ones(len(Rc.row),np.float32),(Rc.row,np.arange(len(Rc.row)))),shape=(Rm.shape[0],len(Rc.row)))
    A=(S@O).reshape(-1,d,d)*alpha+YtY[None]+lam*np.eye(d)[None]
    b=(1+alpha)*(Rm@Y)
    return np.linalg.solve(A,b[...,None])[...,0]
rng=np.random.default_rng(0)
for d in (8,12,16):
  for alpha in (5.,20.):
    Y=rng.normal(0,0.1,(NI,d)).astype(np.float64)
    for it in range(10):
        X=side(R,Y,alpha,1.0); Y=side(R.T.tocsr(),X,alpha,1.0)
    ev(cosf(Y),f'ALS d{d} a{alpha} item-cos',res=(d==12))
# 1(d) per-user ridge on fixed SVD item vectors, using the user's pre-C impressions (clicked vs shown)
T=np.load('/tmp/dig/tr_ev.npz'); Vt=np.load('/tmp/dig/val_t.npz')
W=nrm(np.load('V_ab55.npy'))
su=np.unique(Vt['evuser']); m=np.isin(T['user'],su)
us=np.repeat(T['user'][m],10); ci=T['cand'][m].ravel(); yy=(np.arange(10)[None,:]==T['lab'][m][:,None]).ravel().astype(float)
order=np.argsort(us,kind='stable'); us,ci,yy=us[order],ci[order],yy[order]
uu,st=np.unique(us,return_index=True); en=np.r_[st[1:],len(us)]
ur={}
for lam in (1.,10.):
    for u0,a0,b0 in zip(uu,st,en):
        X=W[ci[a0:b0]]; t=yy[a0:b0]-yy[a0:b0].mean()
        ur[u0]=np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@t)
    Uv=np.array([ur.get(u,np.zeros(W.shape[1])) for u in Vt['evuser']])
    xr=(Uv[Vt['evix']]*W[Vt['cix']]).sum(1)
    ev(xr,f'per-user ridge lam{lam} (imps+clicks)',res=False)
xc=cosf(np.load('V_ab55.npy')); ev(xc+0*xr,'history centroid cos (same V)',res=False)
print('stack centroid+ridge',resid(np.c_[xc,xr],2),' centroid alone',resid(xc,2))
