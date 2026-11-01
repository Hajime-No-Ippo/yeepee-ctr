import sys,glob; sys.path.insert(0,'/workspace/huawei/v21seq')
from evalseq import *
import polars as pl
D={}
for c in (23,25):
    y=yv(c); z=zbase('A',c); fp=sorted(glob.glob(f'/workspace/huawei/v21seq/ens/ctl6_A_c{c}_s*.npy')); M=np.mean([tologit(np.load(f)[:,-1]) for f in fp],0)
    f=pl.read_parquet(f'/workspace/huawei/v12/feat_v3/val{c}.parquet',columns=['hist_n_all','us_since_ev','tm_stale']).to_pandas() if False else pl.read_parquet(f'/workspace/huawei/v12/feat_v3/val{c}.parquet',columns=['hist_n_all','us_since_ev','tm_stale'])
    D[c]=(y,z,M,f)
    print(c,f.describe().to_pandas().T if False else '', AUC(y,.4*z+.6*M))
def seg(f,kind):
    h=f['hist_n_all'].to_numpy(); s=f['tm_stale'].to_numpy(); e=f['us_since_ev'].to_numpy()
    if kind=='hist': return np.digitize(h,[0.5,5,20,60,150])
    if kind=='stale': return np.digitize(s,[1,6,24,72]) if np.nanmax(s)>100 else np.digitize(s,np.nanquantile(s,[.25,.5,.75]))
    if kind=='since': return np.digitize(np.nan_to_num(e),np.nanquantile(e,[.2,.4,.6,.8]))
ws=[.2,.4,.5,.6,.7,.8,.9]
for kind in ('hist','stale','since'):
    res={}
    for c,(y,z,M,f) in D.items():
        g=seg(f,kind); K=g.max()+1; base=AUC(y,.4*z+.6*M); w=np.full(len(y),.6); wk=[]
        for k in range(K):  # coordinate: choose w per segment on this cut
            best=None
            for ww in ws:
                w2=w.copy(); w2[g==k]=ww; a=AUC(y,(1-w2)*z+w2*M)
                if best is None or a>best[0]: best=(a,ww)
            w[g==k]=best[1]; wk.append(best[1])
        res[c]=(wk,AUC(y,(1-w)*z+w*M)-base,np.bincount(g,minlength=K))
        print(kind,c,'seg weights',wk,'in-sample gain',round(res[c][1],5),'sizes',res[c][2])
    # cross apply
    for a_,b_ in ((23,25),(25,23)):
        y,z,M,f=D[b_]; g=seg(f,kind); wk=res[a_][0]; w=np.array(wk)[g]; print('  weights from',a_,'applied to',b_,'gain',round(AUC(y,(1-w)*z+w*M)-AUC(y,.4*z+.6*M),5))
