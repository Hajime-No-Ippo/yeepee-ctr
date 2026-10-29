import sys,time,json; sys.path.insert(0,'/workspace/huawei/dw_p33')
from p33lib import *
D=Data(); c=25; X=Jd(c); V=D.val(c); S=D.snap(X); y=V['y']
ev=V['e'][::10]; it=V['item']; n=len(ev)
def ev_auc(sc):
    a=AUC(y,np.nan_to_num(sc)); m=sc.reshape(-1,10); z=((m-m.mean(1,keepdims=True))).ravel(); return round(a,4),round(AUC(y,z),4)
def score(H,E,mode):
    UV=H@E   # n x k
    if mode=='dot': return (UV[np.repeat(np.arange(n),10)]*E[it]).sum(1)
    if mode=='cos':
        Un=UV/(np.linalg.norm(UV,axis=1,keepdims=True)+1e-9); En=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
        return (Un[np.repeat(np.arange(n),10)]*En[it]).sum(1)
    if mode=='ucdot':  # user vec raw, cand normalised
        En=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9); return (UV[np.repeat(np.arange(n),10)]*En[it]).sum(1)
res=[]
Hs={tau:hist_weights(D,ev,tau,'exp') for tau in (None,10.0,3.0)}
for (W,wm) in ((5,'inv'),(20,'inv'),(60,'one')):
    t=time.time(); M=click_pairs_matrix(D,S,W,wm); P=ppmi(M); Uf,s=svd_embed(P,256); print('W',W,wm,'nnz',M.nnz,P.nnz,'sv',np.round(s[[0,1,2,5,11,15,19,31,63,127,255]],2),round(time.time()-t),'s',flush=True)
    np.save(f'sv_ppmi_W{W}{wm}.npy',s)
    for k in (8,12,16,24,32,64,128,256):
        for p in (0.5,):
            E=emb(Uf,s,k,p)
            for tau in (None,3.0):
                for mode in ('dot','cos','ucdot'):
                    a=ev_auc(score(Hs[tau],E,mode)); r=dict(W=W,wm=wm,k=k,p=p,tau=tau,mode=mode,auc=a[0],auc_we=a[1]); res.append(r); print(json.dumps(r),flush=True)
open('t2_probe.json','w').write(json.dumps(res))
