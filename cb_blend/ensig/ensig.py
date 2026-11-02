"""ensig.py -- diagnostic scan: orthogonality of every eval-side member vs EN and S1 (+ local held-out gain over eq4+graph).
READ-ONLY on all other folders. No eval labels exist; only rank statistics of prediction vectors on eval rows (no training, no features).
Global rank/N only. CPU 4 threads (cores 0-3). Writes only into cb_blend/ensig/."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import sys, json, glob, zipfile, io
import numpy as np, polars as pl
from scipy.stats import rankdata, spearmanr, norm
from sklearn.metrics import roc_auc_score
H='/workspace/huawei/'; OUT=H+'cb_blend/ensig/'
nr=lambda x:((rankdata(x)-1)/(len(x)-1)).astype(np.float64)
def tologit(lp): lp=np.minimum(np.asarray(lp,np.float64),-1e-6); return lp-np.log1p(-np.exp(lp))
mrow=pl.read_parquet(H+'erbrown/merged.parquet',columns=['rowid'])['rowid'].to_list()
rid=pl.read_parquet(H+'v12/feat_v3/eval.parquet',columns=['rowid'])['rowid'].to_list()
ev=pl.read_csv(H+'data/evaluation.csv',columns=['rowid'])['rowid'].to_list()
assert rid==ev and mrow==ev           # eval.parquet == evaluation.csv == merged.parquet row order (checked)
N=len(mrow); assert N==529390
EN=np.load(H+'erbrown/w10an/eb_new_rank.npy').astype(np.float64)
# ---- S1 rebuilt read-only from mkS.py recipe ----
Mdf=pl.read_parquet(H+'erbrown/merged.parquet'); E0=nr(Mdf['eb'].to_numpy()); Pq=nr(Mdf['p33_rank'].to_numpy()); Oo=nr(Mdf['ours'].to_numpy())
A_=np.load(H+'v21fit/A_all.npy').astype(np.float64); D_=np.load(H+'v21fit/D_dcn.npy').astype(np.float64)
V8=np.load(H+'v21fit/inv_v8regen.npy').astype(np.float64); V5=np.load(H+'v21fit/inv_v5.npy').astype(np.float64)
c,g,q,v8,v5=np.load(H+'v21fit/pS1.npy'); S1=nr(EN-c*E0+g*(1-q)*A_+g*q*D_+v8*V8+v5*V5)
z=zipfile.ZipFile(H+'v21fit/out/probe_v21fit_S1_R1step_c109_g123_q67_v8m052_v5m033.csv.zip'); d=pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))
assert d['rowid'].to_list()==mrow; S1file=nr(d['predicted_probability'].to_numpy()); print('S1 rebuilt vs S1 file spearman',spearmanr(S1[::3],S1file[::3])[0],'maxabs',np.abs(S1-S1file).max())
S1=S1file   # use the uploaded file as ground truth
# ---------------- eval-side loaders: all return vectors in merged order ----------------
def ld_seeds(pat,col=-1):                      # c27 files in eval.parquet order == merged order
    fs=sorted(glob.glob(H+pat)); assert fs,pat; P=[]
    for f in fs:
        a=np.load(f); a=a[:,col] if a.ndim>1 else a; a=a.astype(np.float64)
        P.append(tologit(a) if a.max()<=0 else a)
    return np.mean(P,0),len(fs)
def ld_one(f): a=np.load(H+f); return a.astype(np.float64)
def ld_rowid(arrf,ridf):
    a=np.load(H+arrf).astype(np.float64); r=np.load(H+ridf,allow_pickle=True).astype(str)
    return pl.DataFrame({'rowid':r,'p':a}).join(pl.DataFrame({'rowid':mrow}),on='rowid',how='right',maintain_order='right')['p'].to_numpy()
# registry: name -> dict(family, eval=(callable->vec,nseeds,path), local={cut: (pattern with {c}, col)})   local patterns: seed-glob, last col
def L(p,col=-1): return (p,col)
REG={
 # --- our v21 family (in S1 via A_all and D) ---
 'ctl6(M)':   dict(fam='listwise MLP (v21seq)',ev=lambda:ld_seeds('v21seq/ens/ctl6_A_c27_s*.npy'),loc={c:L(f'v21seq/ens/ctl6_A_c{c}_s*.npy') for c in (23,25,26)},ins1='via A'),
 'big(v21seq)':dict(fam='trf history/DIN (v21seq)',ev=lambda:ld_seeds('v21seq/ens/v21seq_big_A_c27_s*.npy'),loc={c:L(f'v21seq/ens/v21seq_big_A_c{c}_s*.npy') for c in (23,25,26)},ins1='via A'),
 'A_all':     dict(fam='ctl6+big',ev=lambda:A_,loc={},ins1='YES'),
 'D_dcn':     dict(fam='DCN-v2',ev=lambda:D_,loc={},ins1='YES'),
 'list(v21list)':dict(fam='listwise MLP (v21list)',ev=lambda:ld_seeds('v21diag/ens/v21listA_c27_s*.npy'),loc={c:L(f'v21diag/ens/v21listA_c{c}_s*.npy') for c in (23,25)},ins1='no (~=A rho .9995)'),
 'graph7s':   dict(fam='LightGCN graph (v21graph)',ev=lambda:ld_rowid('v21graph/out/graph_eval_v21seqorder.npy','v21graph/out/graph_eval_rowid.npy'),loc={c:('GRAPH',c) for c in (23,25,26)},ins1='no (G2 = S1+.05G)'),
 # --- GBDT stack family z (v19 nmA, LightGBM + slate MLP) ---
 'z_lgb(nmAs)':dict(fam='LightGBM single (v19 nmA)',ev=lambda:ld_seeds('v19/ens/nmAs_c27_s0.npy'),loc={c:L(f'v19/ens/nmAs_c{c}_s0.npy') for c in (23,25)},ins1='no'),
 'z_mlp(nmAsmlp)':dict(fam='MLP 3 seeds (v19 nmA)',ev=lambda:ld_seeds('v19/ens/nmAsmlp_c27_s*.npy'),loc={c:L(f'v19/ens/nmAsmlp_c{c}_s*.npy') for c in (25,)},ins1='no'),
 'dwbig':     dict(fam='LightGBM big (v12)',ev=lambda:ld_seeds('v12/ens/dwbig_c27_s0.npy'),loc={c:L(f'v12/ens/dwbig_c{c}_s0.npy') for c in (23,25,26)},ins1='no'),
 'dw_s':      dict(fam='LightGBM slate (v12)',ev=lambda:ld_seeds('v12/ens/dw_s_c27_s0.npy'),loc={c:L(f'v12/ens/dw_s_c{c}_s0.npy') for c in (23,25,26)},ins1='no'),
 'dwmlp':     dict(fam='MLP (v12)',ev=lambda:ld_seeds('v12/ens/dwmlp_c27_s*.npy'),loc={c:L(f'v12/ens/dwmlp_c{c}_s*.npy') for c in (23,25,26)},ins1='no'),
 'mlp2':      dict(fam='MLP2 (v12)',ev=lambda:ld_seeds('v12/ens/mlp2_c27_s*.npy'),loc={c:L(f'v12/ens/mlp2_c{c}_s*.npy') for c in (23,25,26)},ins1='no'),
 'sCbig':     dict(fam='LightGBM v11',ev=lambda:ld_seeds('v12/ens/sCbig_c27_s0.npy'),loc={c:L(f'v12/ens/sCbig_c{c}_s0.npy') for c in (23,25,26)},ins1='no'),
 'coAp':      dict(fam='co-occurrence LGB (v17cooc)',ev=lambda:ld_seeds('v17cooc/ens/coAp_c27_s0.npy'),loc={c:L(f'v17cooc/ens/coAp_c{c}_s0.npy') for c in (23,25,26)},ins1='no'),
 'coBp':      dict(fam='co-occurrence LGB B (v17cooc, transductive)',ev=lambda:ld_seeds('v17cooc/ens/coBp_c27_s0.npy'),loc={c:L(f'v17cooc/ens/coBp_c{c}_s0.npy') for c in (23,25,26)},ins1='no'),
 'coAmlp':    dict(fam='co-occurrence MLP (v17cooc)',ev=lambda:ld_seeds('v17cooc/ens/coAmlp_c27_s*.npy'),loc={c:L(f'v17cooc/ens/coAmlp_c{c}_s*.npy') for c in (25,)},ins1='no'),
 'xgb(v12x)': dict(fam='XGBoost (v12x)',ev=lambda:ld_seeds('v12x/ens/xgb_c27_s0.npy'),loc={c:L(f'v12x/ens/xgb_c{c}_s0.npy') for c in (23,25)},ins1='no'),
 'lr(v9d)':   dict(fam='logistic regression (v9d)',ev=lambda:ld_seeds('v9d/val/ens/lr_c27.npy'),loc={c:L(f'v9d/val/ens/lr_c{c}.npy') for c in (23,25,26)},ins1='no'),
 'ctl_b(v9d)':dict(fam='old MLP (v9d)',ev=lambda:ld_seeds('v9d/val/ens/ctl_b_c27_s0.npy'),loc={c:L(f'v9d/val/ens/ctl_b_c{c}_s0.npy') for c in (25,)},ins1='no'),
 'v7base':    dict(fam='LightGBM v7 (v2work)',ev=lambda:ld_seeds('v2work/ens/v7base_c27_s*.npy'),loc={c:L(f'v2work/ens/v7base_c{c}_s*.npy') for c in (25,26)},ins1='no'),
 'v6base':    dict(fam='LightGBM v6 (v2work)',ev=lambda:ld_seeds('v2work/ens/v6base_c27_s*.npy'),loc={c:L(f'v2work/ens/v6base_c{c}_s*.npy') for c in (25,26)},ins1='no'),
 'v2base':    dict(fam='LightGBM v2 (v2work, ne cleared)',ev=lambda:ld_seeds('v2work/ens/base_c27_s*.npy'),loc={c:L(f'v2work/ens/base_c{c}_s*.npy') for c in (25,26)},ins1='no'),
 'v2slow':    dict(fam='LightGBM v2 slow (v2work)',ev=lambda:ld_seeds('v2work/ens/slow_c27_s0.npy'),loc={c:L(f'v2work/ens/slow_c{c}_s0.npy') for c in (25,26)},ins1='no'),
 'estrc':     dict(fam='v2work estrc',ev=lambda:ld_seeds('v2work/ens/estrc_c27_s*.npy'),loc={c:L(f'v2work/ens/estrc_c{c}_s0.npy') for c in (25,26)},ins1='no'),
 'wide':      dict(fam='v2work wide',ev=lambda:ld_seeds('v2work/ens/wide_c27_s*.npy'),loc={c:L(f'v2work/ens/wide_c{c}_s0.npy') for c in (25,26)},ins1='no'),
 # --- eval-only (no fold predictions) ---
 'EN':        dict(fam='ErBrown eval-only',ev=lambda:EN,loc={},ins1='base'),
 'E0':        dict(fam='ErBrown old (submission_best)',ev=lambda:E0,loc={},ins1='YES (-.109)'),
 'p33':       dict(fam='teammate p33',ev=lambda:Pq,loc={},ins1='no'),
 'v8regen':   dict(fam='v8 regen (xgb8)',ev=lambda:V8,loc={},ins1='YES (-.052)'),
 'v5':        dict(fam='v5 submission (LGB, early)',ev=lambda:V5,loc={},ins1='YES (-.033)'),
 'EBDISTILL': dict(fam='ctl6 distilled on EN labels',ev=lambda:nr(np.load(H+'v21fit/S_ebdistill.npy').astype(np.float64)),loc={},ins1='no'),
 'ours_O(v21list B)':dict(fam='ErBrown-side merged O',ev=lambda:Oo,loc={},ins1='no'),
}
