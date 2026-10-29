"""Leak tests for v18 families (val25 and val23): (a) rebuild == stored; (b) flip ALL labels of events with ts>=X -> identical;
(c) corrupt every NON-target event with ts>=X (random slate items, random users, flipped labels, random history items/timestamps) -> identical
(own event's slate/history of the target rows are legitimately used)."""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/workspace/huawei/v18classic')
from mkv18 import *
D=Data()
for nm in ('val25','val23'):
    X,V=rows_for(D,nm); nX=int(np.searchsorted(D.ETS,X)); R0=build(D,X,V)
    ok=all(np.allclose(R0[f][k],pl.read_parquet(f'{O}/feat_{f}/{nm}.parquet')[k].to_numpy(),equal_nan=True,atol=1e-5) for f in R0 for k in R0[f]); print(nm,'(a) matches stored:',ok)
    Y0=D.Y.copy(); D.Y[nX:]=1-D.Y[nX:]; R1=build(D,X,V); D.Y[:]=Y0
    same=lambda A,B:all(np.array_equal(A[f][k],B[f][k],equal_nan=True) for f in A for k in A[f]); print(nm,'(b) flip all post-cutoff labels -> identical:',same(R0,R1))
    rng=np.random.default_rng(0); tgt=np.zeros(D.NE,bool); tgt[V['e']]=True; bad=(np.arange(D.NE)>=nX)&~tgt
    sv=(D.Y.copy(),D.RI.copy(),D.EU.copy(),D.HI.copy(),D.HT.copy())
    D.Y[bad]=1-D.Y[bad]; D.RI[bad]=rng.integers(0,D.NI,(bad.sum(),10)); D.EU[bad]=rng.integers(0,D.NU,bad.sum())
    hb=np.zeros(len(D.HI),bool); evi,idx,_=flat(D,np.where(bad)[0]); hb[idx]=True; D.HI[hb]=rng.integers(0,D.NI,hb.sum()); D.HT[hb]=rng.integers(J20,J20+7*DAY,hb.sum())
    R2=build(D,X,V); D.Y,D.RI,D.EU,D.HI,D.HT=sv
    print(nm,f'(c) corrupt {int(bad.sum())} non-target post-cutoff events ({int(hb.sum())} history entries) -> identical:',same(R0,R2))
