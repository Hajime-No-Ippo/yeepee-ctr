exec(open('svdvar6.py').read().split('for N in')[0])
from sklearn.utils.extmath import randomized_svd
dfv=A.filter(pl.col('L')-pl.col('p')<=30).unique(['u','ix'])
M0=sp.csr_matrix((np.ones(dfv.height,np.float32),(dfv['u'].to_numpy(),dfv['ix'].to_numpy())),shape=(LH.height,NI))
pop=np.asarray(M0.sum(0)).ravel()
def W(V): 
    x,_=cosf(V); dd=D.with_columns(pl.Series('x',x)); return {c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}
Ma=M0.multiply(1/np.sqrt(pop+5)).tocsr()
for it in (5,20):
    U,S,Vt=randomized_svd(Ma,12,random_state=0,n_iter=it); print('alpha.5 randomized k12 n_iter',it,W((Vt.T*S)[:,1:]))
idf=np.log((LH.height+1)/(pop+1)); Mi=M0@sp.diags(idf); Mi=sp.diags(1/(np.sqrt(np.asarray(Mi.multiply(Mi).sum(1)).ravel())+1e-6))@Mi
for k in (12,16,32):
    _,s,vt=svds(Mi.tocsr(),k=k,random_state=0); o=np.argsort(-s); s=s[o]; vt=vt[o]
    print('idf+rownorm svds k',k,'keep1',W(vt.T*s),'drop1',W(vt[1:].T*s[1:]), 'unscaled', W(vt.T))
U,S,Vt=randomized_svd(Mi,64,random_state=0,n_iter=4); print('existing-style: idf+rownorm, rsvd k64 n_iter4, first16 unscaled',W(Vt[:16].T),' first12',W(Vt[:12].T))
