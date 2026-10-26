exec(open('svdvar.py').read().split('res={}')[0])
v=np.ones(cnt.height,np.float32)
M0=sp.csr_matrix((v,(cnt['u'].to_numpy(),cnt['ix'].to_numpy())),shape=(cnt['u'].max()+1,NI))
pop=np.asarray(M0.sum(0)).ravel()
Mn=M0.multiply(1/(pop+5)**0.5).tocsr()
_,s,vt=svds(Mn,k=40,random_state=0); o=np.argsort(-s); s=s[o]; vt=vt[o]
def cosf(V):
    V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9)
    hc=np.zeros((H['k'].max()+1,V.shape[1])); np.add.at(hc,H['k'].to_numpy(),V[H['ix'].to_numpy()])
    hcr=np.where(rk[:,None]>=0,hc[np.maximum(rk,0)],0)
    return (V[ix]*hcr).sum(1)/(np.linalg.norm(hcr,axis=1)+1e-9)
cands={}
for kk in (12,13,14,15): cands[f'hard_k{kk}']=vt[:kk].T*s[:kk]
for kk in (12,13): cands[f'hard_k{kk}_nopop']=vt[1:kk].T*s[1:kk]
for fl in (1.70,1.72,1.75):
    w=np.maximum(s-fl,0); cands[f'shrink{fl}']=vt.T*w
    cands[f'shrink{fl}_nopop']=(vt.T*w)[:,1:]
for n,V in cands.items():
    x=cosf(V); dd=D.with_columns(pl.Series('x',x))
    sa={c:round(roc_auc_score(y[cut==c],x[cut==c]),4) for c in (25,26)}
    wa={c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}
    r=resid(x); print(f'{n:20s}','glob',sa,'within',wa,'resid',r,flush=True); np.save(f'x_{n}.npy',x)
