exec(open('svdvar.py').read().split('D=pl.read_parquet')[0])
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
v=np.ones(cnt.height,np.float32)
M0=sp.csr_matrix((v,(cnt['u'].to_numpy(),cnt['ix'].to_numpy())),shape=(cnt['u'].max()+1,NI))
pop=np.asarray(M0.sum(0)).ravel()
_,s,vt=svds(M0.multiply(1/(pop+5)**0.5).tocsr(),k=13,random_state=0); o=np.argsort(-s); V=(vt[o].T*s[o])[:,1:]
keep=pop>=5; print("kept",keep.sum()); Vn=V[keep]/np.linalg.norm(V[keep],axis=1,keepdims=True)
rng=np.random.default_rng(0); sub=rng.choice(len(Vn),8000,replace=False)
for k in (4,8,10,12,13,16,20,24,32):
    km=KMeans(k,n_init=3,random_state=0).fit(Vn); print(k,'inertia',round(km.inertia_,1),'sil',round(silhouette_score(Vn[sub],km.labels_[sub]),3),flush=True)
# distribution of max abs cosine to nearest of the 12 axes after rotation? check norm distribution
print('row norms (pre-normalise) quantiles',np.quantile(np.linalg.norm(V[keep],axis=1),[.05,.25,.5,.75,.95]).round(3))
np.save('V12.npy',V)
