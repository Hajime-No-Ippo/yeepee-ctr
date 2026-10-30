import sys,time; sys.path.insert(0,'/workspace/huawei/v17cooc'); sys.dont_write_bytecode=True
from mkcooc import *
from sklearn.decomposition import NMF
D=Data(); X=CUT['val25']; S=D.snap(X)
t=time.time(); P=ppmi(click_pairs_matrix(D,S,20,'inv'),1.0,0.0); print('ppmi',P.shape,P.nnz,round(time.time()-t),flush=True)
for sol in ('mu',):
    t=time.time(); m=NMF(12,init='nndsvda',solver=sol,max_iter=60,random_state=0,tol=1e-3); W=m.fit_transform(P); print('nmfP',sol,round(time.time()-t),'s',m.n_iter_,W.shape,np.round(W.sum(0)[:12],0),flush=True)
