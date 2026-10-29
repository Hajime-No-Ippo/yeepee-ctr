import sys,json; sys.path.insert(0,'/workspace/huawei/dw_p33')
from p33lib import *
D=Data(); X=Jd(25); V=D.val(25); S=D.snap(X); y=V['y']; ev=V['e'][::10]; it=V['item']; n=len(ev); rep=np.repeat(np.arange(n),10)
def au(sc): 
    m=sc.reshape(-1,10); return round(AUC(y,np.nan_to_num(sc)),4),round(AUC(y,(m-m.mean(1,keepdims=True)).ravel()),4)
def cosr(H,E):
    UV=H@E; Un=UV/(np.linalg.norm(UV,axis=1,keepdims=True)+1e-9); En=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9); return (Un[rep]*En[it]).sum(1)
H0=hist_weights(D,ev,None)
for W,wm,al,sh in ((20,'inv',0.75,0),(20,'inv',1.0,0),(20,'inv',0.75,1.0),(20,'one',0.75,0)):
    M=click_pairs_matrix(D,S,W,wm); P=ppmi(M,al,sh); U,s=svd_embed(P,64)
    for lo,hi in ((0,12),(0,13),(1,13),(1,14),(1,17)):
        for p in (0.0,0.5,1.0):
            E=(U[:,lo:hi]*(s[lo:hi]**p)); print(W,wm,'alpha',al,'shift',sh,'comps',lo,hi,'p',p,au(cosr(H0,E)),flush=True)
