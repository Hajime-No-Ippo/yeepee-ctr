import sys,time; sys.path.insert(0,'/workspace/huawei/v17cooc'); sys.dont_write_bytecode=True
from mkcooc import *
from scipy import sparse
D=Data(); n_orig=len(D.uu); X=CUT['val25']; V=rows_train(D,'val25'); ev=V['e'][::10]
S=D.snap(X); print({k:(len(v) if hasattr(v,'__len__') else v) for k,v in S.items()})
S2,nadd=extend_S(D,S,ev,n_orig); print('B',len(S2['cu']),nadd,'users',len(np.unique(S2['cu'])),'items',len(np.unique(S2['ci'])),'NI',D.NI,'NU',D.NU)
for nm,SS in (('A',S),('B',S2)):
    uq,ui=np.unique(SS['cu'],return_inverse=True)
    M=sparse.csr_matrix((np.ones(len(ui),np.float32),(ui,SS['ci'])),shape=(len(uq),D.NI)); M.sum_duplicates(); print(nm,M.shape,M.nnz)
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    for sol in ('cd','mu'):
        t=time.time(); m=NMF(12,init='nndsvda',solver=sol,max_iter=60 if sol=='cd' else 100,random_state=0,tol=1e-3); W=m.fit_transform(M); print(nm,'nmf',sol,round(time.time()-t),'s err',m.reconstruction_err_,m.n_iter_,flush=True)
    t=time.time(); lda=LatentDirichletAllocation(12,learning_method='batch',max_iter=15,random_state=0,n_jobs=8,evaluate_every=-1); Wl=lda.fit_transform(M); print(nm,'lda',round(time.time()-t),'s',flush=True)
    break
