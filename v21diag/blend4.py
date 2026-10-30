import os; os.environ['OMP_NUM_THREADS']='2'
import sys, json, numpy as np
sys.path.insert(0,'.'); from diag1 import *
from combine3 import comb
from scipy.special import logsumexp
def lst(tag,cut,seeds,col=3):
    return np.mean([np.load(f'ens/{tag}_c{cut}_s{s}.npy')[:,col] for s in seeds],0)
def tologit(lp): lp=np.minimum(lp,-1e-6); return lp-np.log1p(-np.exp(lp))
def stats(z,y,name):
    Z=z.reshape(-1,10); has=y.reshape(-1,10).max(1); p=expit(Z)
    return dict(name=name,glob=round(AUC(y,z),5),within=round(within_auc(z,y),5),slate_sump=round(AUC(has,p.sum(1)),4),slate_lse=round(AUC(has,logsumexp(Z,1)),4))
