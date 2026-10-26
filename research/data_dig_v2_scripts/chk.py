import numpy as np, polars as pl, sys; sys.path.insert(0,'/tmp/dig')
exec(open('/tmp/dig/lat_ev.py').read())
V=np.load('val_t.npz'); W=np.load('V_ab55.npy'); W=W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-9)
H=V['H']; m=H>=0; u=(W[np.maximum(H,0)]*m[...,None]).sum(1)
q((u[V['evix']]*W[V['cix']]).sum(1),'svd dot via val_t')
T=np.load('tr_ev.npz'); 
# train-side: within-event AUC of the svd score on last day of pre-C train (J24) events
sel=T['ts']>=1737331200+4*86400; H=T['H'][sel]; m=H>=0; u=(W[np.maximum(H,0)]*m[...,None]).sum(1)
sc=(u[:,None,:]*W[T['cand'][sel]]).sum(-1); lab=T['lab'][sel]; c=lab<10
r=(sc[c]>sc[c,lab[c]][:,None]).sum(1)+0.5*((sc[c]==sc[c,lab[c]][:,None]).sum(1)-1)
print('J24 train within-AUC of svd', 1-r.mean()/9)
