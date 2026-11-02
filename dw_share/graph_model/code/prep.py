import sys,time,os; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/v12'); os.environ['PYTHONDONTWRITEBYTECODE']='1'
from mkdw import *     # Data, CUT, rows_train, extend_eval, J20, DAY, Jd (chdir to dw_signals)
O='/workspace/huawei/v21graph/out/'
D=Data(); NU,NI=D.NU,D.NI; print(NU,NI,flush=True)
HL=30
def hist(ev):
    n=len(ev); hi=np.full((n,HL),NI,np.int32); ht=np.zeros((n,HL),np.int64)
    for j,e in enumerate(ev):
        a,b=D.OFF[e],D.OFF[e+1]; it=D.HI[a:b]; t=D.HT[a:b]; o=np.argsort(-t,kind='stable')[:HL]; hi[j,:len(o)]=it[o]; ht[j,:len(o)]=t[o]
    return hi,ht
seq='/workspace/huawei/v21seq/seqfeat/'
def tab(nm,V,y=None):
    ev=V['e'][::10]; hi,ht=hist(ev)
    s=np.load(seq+f'A_{nm}.npz',allow_pickle=True)
    assert (np.asarray(s['rowid']).astype(str)==np.asarray(V['rowid']).astype(str)).all(),nm
    ets=D.ETS[ev]
    np.savez(O+f'tab_{nm}.npz',rowid=np.asarray(V['rowid']).astype(str),user=V['user'].astype(np.int32),item=V['item'].astype(np.int32),ts=ets,hist=hi,hts=ht,
             y=(y if y is not None else np.zeros(len(V['rowid']),np.int8)))
    print('tab',nm,len(ev),flush=True)
for nm in ['f1','f2','f3','f4','f5','f6','val23','val25','val26']:
    V=rows_train(D,nm); y=pl.read_parquet(f'{F3}/{nm}.parquet',columns=['clicked'])['clicked'].to_numpy().astype(np.int8)
    tab(nm,V,y)
# graphs (train-only) at cutoffs
for k in range(1,8):
    S=D.snap(J20+k*DAY); np.savez(O+f'g_{k}.npz',cu=S['cu'],ci=S['ci'],ct=S['ct'],nX=S['nX']); print('g',k,len(S['cu']),flush=True)
V=extend_eval(D); tab('eval',V)
np.save(O+'dims.npy',np.array([NU,NI]))
