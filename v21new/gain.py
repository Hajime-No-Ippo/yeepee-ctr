import sys,numpy as np; sys.dont_write_bytecode=True
from sklearn.metrics import roc_auc_score as AUC
from scipy.stats import rankdata
CB='/workspace/huawei/cb_blend/'; G8='/workspace/huawei/v21graph2/out/graph2s3b_f6_c%d.npy'
def nr(x): return rankdata(x)/len(x)
def load(c):
    d=np.load(CB+f'oof_c{c}.npz',allow_pickle=True); g=np.load(G8%c)
    return d,g
def bases(c,d):
    nz=lambda k:nr(d[k])
    B={'base':.4*nz('raw_z')+.6*nz('raw_M') if 'raw_z' in d.files else nz('raw_M')}
    if 'raw_D' in d.files:
        B['stackD']=.3*nz('raw_z')+.3*nz('raw_M')+.4*nz('raw_D'); B['eq4']=.25*(nz('raw_z')+nz('raw_M')+nz('raw_big')+nz('raw_D'))
    else: B['M.5big']=.5*nz('raw_M')+.5*nz('raw_big')
    return B
def report(c,N,name,WG=(.1,.15,.2,.25,.3),WN=(0,.02,.04,.06,.08,.1,.15,.2,.3)):
    d,g=load(c); y=d['y']; Gn=nr(g); Nn=nr(N); B=bases(c,d)
    print(f'C{c} {name}: standalone N {AUC(y,N):.5f}  G8 {AUC(y,g):.5f}  rho(N,G8) {np.corrcoef(Nn,Gn)[0,1]:.3f} rho(N,M) {np.corrcoef(Nn,nr(d["raw_M"]))[0,1]:.3f}',flush=True)
    res={}
    for bn,b in B.items():
        # base+graph at best wg
        bg=[(AUC(y,(1-w)*b+w*Gn),w) for w in WG]; a0,wg=max(bg); 
        out=[]
        for wn in WN[1:]:
            out.append((AUC(y,(1-wn)*((1-wg)*b+wg*Gn)+wn*Nn)-a0,wn))
        # also re-tune wg jointly small grid
        best=max(out); res[bn]=(a0,wg,best)
        print(f'  {bn:7s} base+G8 {a0:.5f}@wg{wg}; +N best {best[0]*1e5:+.1f}e-5 @wn{best[1]}  | '+' '.join(f'{w}:{g*1e5:+.0f}' for g,w in out),flush=True)
    return res
if __name__=='__main__':
    c=int(sys.argv[1]); N=np.load(sys.argv[2]); report(c,N,sys.argv[2].split('/')[-1])
