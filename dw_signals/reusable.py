"""Clean entry points for the recommended new signals.

All functions take
    D : common.Data                (lite events/rows loader)
    X : fit cutoff, unix seconds   (only train events with ts < X are used for any fit/aggregate)
    rows : dict from D.val(c) / same layout: 'user','item','ts','e' aligned to n_events*10 rows (10 candidates per event)
and return {feature_name: float32 array of len(rows['item'])} (NaN = no information for that row).

Leak-safety: every statistic (impression matrix, SVDs, click log) is built from events with ts < X only; the only
per-row inputs are the row's own user id, candidate item, ts and the row's OWN pre-ts history (D.HI/HT/OFF). No eval-row
labels, impressions or histories are ever aggregated. Rows' within-slate stats (centrality) use only the row's own slate.

    from reusable import impression_taste, slate_centrality, skip_contrast, all_recommended
"""
import numpy as np, polars as pl
from common import *
import signals as S_

def impression_taste(D,X,rows,ks=(8,16,32,64,128)):
    """#1 Impression-SVD taste. SVD(k) of the user x item IMPRESSION-count matrix (train events ts<X, all 10 shown items per event,
    clicked or not; scaled 1/sqrt((nu+1)(ni+1))). Features per k: cos(user vec, item vec), cos(centroid of the row's own click-history items in this
    item space, candidate), + within-slate z-scores. Also skip-only SVD(32) (shown-but-not-clicked matrix) and max-sim to history at k=64."""
    snap=D.snap(X); out={}
    out.update(S_.sig_impression_svd(D,snap,rows,X)[0])             # k=32 base (cos, reconstruction, z, rank)
    out.update(S_.sig_impression_variants(D,snap,rows,X)[0])        # k=16,64, skip-svd32, joint click+imp svd, hist centroid in imp space
    out.update(S_.sig_impression_multirank(D,snap,rows,X,ks=tuple(k for k in ks if k in (8,64,128)))[0])
    return out

def slate_centrality(D,X,rows):
    """#2 Slate centrality: candidate's mean/max/top3 cosine to the 9 other candidates of the SAME slate, in click-SVD(32) and impression-SVD(32) item spaces,
    plus slate coherence (mean pairwise cos). Captures 'retrieved-cluster membership' / odd-one-out."""
    snap=D.snap(X); Vclk,_=click_svd(snap,D,32); out={}
    out.update(S_.sig_slate_centrality(D,snap,rows,X,Vclk,'clk')[0])
    Vi=S_.sig_impression_svd(D,snap,rows,X)[1]['Vi']; out.update(S_.sig_slate_centrality(D,snap,rows,X,norm(Vi),'imp')[0])
    return out

def skip_contrast(D,X,rows):
    """#3 Shown-but-not-clicked centroid in click-SVD space (all train / last 3d), all-impression centroid, own-history centroid, and history-minus-skip contrasts. (weak alone, small positive in the group)"""
    snap=D.snap(X); Vclk,_=click_svd(snap,D,32); return S_.sig_skip_embedding(D,snap,rows,X,Vclk)[0]

def all_recommended(D,X,rows)->pl.DataFrame:
    F={}; F.update(impression_taste(D,X,rows)); F.update(slate_centrality(D,X,rows)); F.update(skip_contrast(D,X,rows))
    return pl.DataFrame({k:np.asarray(v,np.float32) for k,v in F.items()})

if __name__=='__main__':
    import sys; c=int(sys.argv[1]); D=Data(); V=D.val(c); df=all_recommended(D,Jd(c),V); print(df.shape); print(df.columns)
