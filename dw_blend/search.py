import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np, polars as pl, json
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
H='/workspace/huawei/'
Y={c:pl.read_parquet(f'{H}v12/feat_v3/val{c}.parquet',columns=['clicked'])['clicked'].to_numpy() for c in (23,25)}
P={c:dict(np.load(f'pool_c{c}.npz')) for c in (23,25)}
names=list(P[23]); 
A=lambda c,x:AUC(Y[c],x)
base=lambda c:0.4*P[c]['z']+0.6*P[c]['M']
b={c:A(c,base(c)) for c in (23,25)}; print('base',b)
for m in names: print(f'{m:9s}',round(A(23,P[23][m]),5),round(A(25,P[25][m]),5),'rho_vs_M',round(np.corrcoef(P[25][m][::7],P[25]['M'][::7])[0,1],4))
# greedy forward selection (equal-weight adds with step weights), objective = mean of delta on both cuts; report both
cur={c:base(c).copy() for c in (23,25)}; sel=[]
for it in range(6):
    best=None
    for m in names:
        for w in (0.1,0.2,0.35):
            d=[A(c,(1-w)*cur[c]+w*P[c][m])-A(c,cur[c]) for c in (23,25)]
            sc=min(d)
            if best is None or sc>best[0]: best=(sc,m,w,d)
    sc,m,w,d=best; print('greedy',it,m,w,[round(x,5) for x in d],flush=True)
    if sc<=0.00003: break
    for c in (23,25): cur[c]=(1-w)*cur[c]+w*P[c][m]
    sel.append((m,w))
print('greedy total',{c:round(A(c,cur[c])-b[c],5) for c in (23,25)})
# signed LR in rank space (fit on one cut, test on other), features = ranks centered
def fit(c,C=1.0):
    X=np.column_stack([P[c][m] for m in names]); X=(X-.5)*4
    lr=LogisticRegression(C=C,max_iter=300); lr.fit(X,Y[c]); return lr.coef_[0]
for C in (0.01,1.0):
  for tr,te in ((23,25),(25,23)):
    w=fit(tr,C); s=np.column_stack([P[te][m] for m in names])@w
    print('signedLR C',C,'train',tr,'test',te,round(A(te,s),5),'base',round(b[te],5),'delta',round(A(te,s)-b[te],5),flush=True)
    if C==1.0: print({n:round(x,2) for n,x in zip(names,w) if abs(x)>.15})
