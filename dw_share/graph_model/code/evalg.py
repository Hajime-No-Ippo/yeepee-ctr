import sys,numpy as np; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v21dcn'); sys.path.insert(0,'/workspace/huawei/v21seq')
import rep
from rep import ref,load,nr,AUC
CUTS={23:'val23',25:'val25',26:'val26'}
def bases(c):
    y,z,M,b=ref(c); b1=.4*z+.6*M; out={'base':b1}
    try: D=nr(load('r0_base',c)[0]); out['stackD']=.3*z+.3*M+.4*D
    except Exception as e: pass
    return y,out
def gains(c,G,ws=(.05,.1,.2,.3)):
    y,B=bases(c); G=nr(G); r={}
    for k,b in B.items():
        a0=AUC(y,b); r[k]=(a0,{w:AUC(y,(1-w)*b+w*G)-a0 for w in ws})
    return y,r
def demean(s): return (s.reshape(-1,10)-s.reshape(-1,10).mean(1,keepdims=True)).ravel()
def report(c,name,G):
    y,r=gains(c,G); rho=np.corrcoef(nr(G),nr(ref(c)[2]))[0,1]
    s=f'C{c} {name}: auc {AUC(y,G):.5f} rho_M {rho:.3f} '+' | '.join(f'{k}: '+' '.join(f'w{w}:{g*1e5:+.0f}' for w,g in v[1].items()) for k,v in r.items()); print(s,flush=True); return s
