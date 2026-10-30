"""assemble.py: local blend AUC (cut 25) and candidate CSV (cut 27) for co-occurrence variants.
usage: python assemble.py base|A|B [--out]   blend = (dwbig + dw_s + 2*mean(mlp x3 seeds)) / 4 on raw logits, then sigmoid.
base = v12 components (v12/ens). A = train-only graph. B = TRANSDUCTIVE (train+eval histories in graph)."""
import sys, os, numpy as np, polars as pl, pandas as pd
from sklearn.metrics import roc_auc_score
H='/workspace/huawei'; v=sys.argv[1]
def comp(c):
    if v=='base': E=f'{H}/v12/ens'; big,s,m=f'dwbig_c{c}_s0',f'dw_s_c{c}_s0',f'dwmlp_c{c}_s%d'
    else: E=f'{H}/v17cooc/ens'; big,s,m=f'co{v}p_c{c}_s0',f'co{v}_s_c{c}_s0',f'co{v}mlp_c{c}_s%d'
    L=lambda f,col:np.load(f'{E}/{f}.npy')[:,col]
    b=L(big,0 if v!='base' else 1); d=L(s,1); ml=np.mean([L(m%k,3) for k in (0,1,2)],0)
    return b,d,ml,(b+d+2*ml)/4
y=pl.read_parquet(f'{H}/v12/feat_v3/val25.parquet',columns=['clicked'])['clicked'].to_numpy()
b,d,ml,z=comp(25)
print(v,'C25 dwbig %.5f dw_s %.5f mlp(3) %.5f blend %.5f'%tuple(roc_auc_score(y,t) for t in (b,d,ml,z)))
if '--out' in sys.argv:
    b,d,ml,z=comp(27); p=1/(1+np.exp(-z))
    m=pl.read_parquet(f'{H}/v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list()
    ev=pl.read_csv(f'{H}/data/evaluation.csv',columns=['rowid'])['rowid'].to_list()
    df=pd.DataFrame({'rowid':m,'predicted_probability':p}).set_index('rowid').loc[ev].reset_index()
    ref=pd.read_csv(f'{H}/v5/submission_v5.csv').rowid
    assert len(df)==529390 and set(df.rowid)==set(ref) and np.isfinite(p).all() and df.predicted_probability.between(0,1).all()
    name={'A':'submission_v17cooc_A_trainonly','B':'submission_v17cooc_B_TRANSDUCTIVE'}[v]
    os.makedirs(f'{H}/v17cooc/out',exist_ok=True); df.to_csv(f'{H}/v17cooc/out/{name}.csv',index=False); print('wrote',name,len(df),df.predicted_probability.describe().to_dict())
