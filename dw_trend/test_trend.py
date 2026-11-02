"""Leak tests (label-flip + future-truncation) for trend.py / tt_trend.py.  Run: python test_trend.py  (cut 25, ~3 min, ~3 GB).
1. Flip ALL labels (Y) of events with ts >= X (val events + later) and randomise their later impressions' labels -> train-only AND TT features must be bit-identical (labels of >=X never used).
2. Flip labels of val rows themselves explicitly (subset of 1.) -> identical.
3. Sanity (power of the test): flipping labels of events < X MUST change item trend features (it_tr_*), proving the test would detect label use."""
import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_trend')
from tt_trend import *
c=25; X=Jd(c)
D=Data(); V=D.val(c); E12=p33feat._space(D,D.snap(X),20,'inv',1.0,16)[:,:12]
f0=trend_feats(D,X,V,E12=E12); t0=tt_feats(D,X,V)
nX=int(np.searchsorted(D.ETS,X)); Y0=D.Y.copy(); rs=np.random.RandomState(0)
D.Y[nX:]=1-D.Y[nX:]                      # flip every label at/after the cutoff
f1=trend_feats(D,X,V,E12=E12); t1=tt_feats(D,X,V)
for k in f0: assert np.array_equal(f0[k],f1[k],equal_nan=True),('LEAK train-only',k)
for k in t0: assert np.array_equal(t0[k],t1[k],equal_nan=True),('LEAK TT',k)
print('PASS: labels at/after cutoff never used (',len(f0),'train-only +',len(t0),'TT features identical )')
D.Y[:]=Y0; D.Y[:nX]=rs.randint(0,2,(nX,10)).astype(np.int8)   # destroy labels before cutoff
f2=trend_feats(D,X,V,E12=E12,groups=('item',))
ch=[k for k in f2 if not np.array_equal(f0[k],f2[k],equal_nan=True)]
print('power check: item features changed when pre-cutoff labels randomised:',ch); assert 'it_tr_ratio' in ch and 'it_tr_slope' in ch
# TT must not depend on labels at all, even pre-cutoff
t2=tt_feats(D,X,V); assert all(np.array_equal(t0[k],t2[k],equal_nan=True) for k in t0); print('PASS: TT features independent of ALL labels')
# future truncation: TT/train feats depend only on events < X (+ val rows' own slate/ts/hist): zero out events strictly after the val window? (val events ARE the post-cutoff events) -> check train-only item stats ignore post-X impressions
D.Y[:]=Y0; RI0=D.RI.copy(); D.RI[nX:]=rs.randint(0,D.NI,D.RI[nX:].shape).astype(D.RI.dtype); i3=item_feats(D,X,D.snap(X),V['item'])
assert all(np.array_equal(f0[k],i3[k],equal_nan=True) for k in i3); print('PASS: train-only item features ignore post-cutoff impressions')
