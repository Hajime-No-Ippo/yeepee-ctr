"""v17cooc blend for one variant. python blend_cooc.py CUT VARIANT   (cwd = work dir; components ens/{V}big, ens/{V}_s, ens/{V}mlp_s0..2)
z = (big@1000 + dw_s@500 + 2 * mean_over_3_seeds(mlp@epoch4)) / 4 (logit blend 1:1:2), p = sigmoid(z).
CUT=27 -> eval_pred_{V}.parquet (rowid, p);  CUT=25 -> local AUC on val25 -> local_auc_c25_{V}.json"""
import sys, json, numpy as np, polars as pl
sys.path.insert(0,'.')
from cl import matrix, BASE
cut=int(sys.argv[1]); V=sys.argv[2]; vn='eval' if cut==27 else f'val{cut}'
_,m,_=matrix(vn,1.0,extra=BASE)
big=np.load(f'ens/{V}big_c{cut}_s0.npy')[:,0].astype(np.float64); sm=np.load(f'ens/{V}_s_c{cut}_s0.npy')[:,0].astype(np.float64)
ml=np.mean([np.load(f'ens/{V}mlp_c{cut}_s{s}.npy')[:,3].astype(np.float64) for s in (0,1,2)],0)
assert len(big)==len(sm)==len(ml)==m.height
z=(big+sm+2*ml)/4; p=1/(1+np.exp(-z)); assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
if cut==27:
    pl.DataFrame({'rowid':m['rowid'],'p':p}).write_parquet(f'eval_pred_{V}.parquet'); print('wrote',f'eval_pred_{V}.parquet',m.height)
else:
    from sklearn.metrics import roc_auc_score
    y=m['clicked'].to_numpy(); r={k:round(roc_auc_score(y,v),5) for k,v in dict(dwbig=big,dw_s=sm,mlp=ml,blend=z).items()}
    print('LOCAL AUC cut',cut,'variant',V,r); json.dump(r,open(f'local_auc_c{cut}_{V}.json','w'))
