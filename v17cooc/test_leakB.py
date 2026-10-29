"""Label-independence test for variant B: rebuild val25 B features (a) as stored, (b) after flipping ALL labels of events with ts>=cutoff -> must be identical."""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/workspace/huawei/v17cooc')
from mkcooc import *
D=Data(); n0=len(D.uu); X=CUT['val25']; V=rows_train(D,'val25')
f0,nadd,_=pp_feats(D,X,V,'B',n0)
st=pl.read_parquet(f'{O}/feat_B/val25.parquet'); print('matches stored feat_B:',all(np.allclose(f0[k],st['coB_'+k].to_numpy(),equal_nan=True,atol=1e-6) for k in f0))
nX=int(np.searchsorted(D.ETS,X)); D.Y[nX:]=1-D.Y[nX:]
f1,_,_=pp_feats(D,X,V,'B',n0)
print('flip all post-cutoff labels -> identical:',all(np.array_equal(f0[k],f1[k],equal_nan=True) for k in f0), 'nadd',nadd)
