exec(open('svdvar.py').read().split('res={}')[0])
# A: full click log per user before C (union of histories), binary
full=EV.select('user','h_items').explode('h_items').unique().collect().rename({'h_items':'item'}).join(imap,on='item')
umap=full.select('user').unique().with_row_index('u'); full=full.join(umap,on='user')
print('clicks per user: last-hist',cnt.height/ (cnt['u'].max()+1),' full log',full.height/umap.height)
def build(pairs_u,pairs_i,nu,a,k):
    M=sp.csr_matrix((np.ones(len(pairs_u),np.float32),(pairs_u,pairs_i)),shape=(nu,NI))
    pop=np.asarray(M.sum(0)).ravel(); M=M.multiply(1/(pop+5)**a).tocsr()
    _,s,vt=svds(M,k=k,random_state=0); o=np.argsort(-s); return s[o],vt[o]
def cosf(V,w=None):
    V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9)
    hc=np.zeros((H['k'].max()+1,V.shape[1])); np.add.at(hc,H['k'].to_numpy(),V[H['ix'].to_numpy()])
    hcr=np.where(rk[:,None]>=0,hc[np.maximum(rk,0)],0)
    return (V[ix]*hcr).sum(1)/(np.linalg.norm(hcr,axis=1)+1e-9), hcr
s,vt=build(full['u'].to_numpy(),full['ix'].to_numpy(),umap.height,0.5,30)
print('full-log spectrum',np.round(s,3))
for kk in (12,13,14,16,18,20):
    x,_=cosf(vt[1:kk].T*s[1:kk]); dd=D.with_columns(pl.Series('x',x))
    sa={c:round(roc_auc_score(y[cut==c],x[cut==c]),4) for c in (25,26)}
    wa={c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}
    print(f'full k{kk} nopop','glob',sa,'within',wa,'resid',resid(x),flush=True); np.save(f'x_full_k{kk}.npy',x)
np.save('full_s.npy',s); np.save('full_vt.npy',vt)
