import sys; sys.path.insert(0,'/workspace/huawei/dw_lb'); sys.dont_write_bytecode=True
from lib import *
import lib
cuts=[int(x) for x in sys.argv[1].split(',')]; groups=sys.argv[2].split(',')
for c in cuts:
    F=load_cut(c); S=pl.read_parquet(f'/workspace/huawei/dw_v12sig/cache/sig_c{c}.parquet').join(pl.read_parquet(f'/workspace/huawei/dw_v12sig/cache/tm_c{c}.parquet'),on='rowid') if os.path.exists(f'/workspace/huawei/dw_v12sig/cache/tm_c{c}.parquet') else pl.read_parquet(f'/workspace/huawei/dw_v12sig/cache/sig_c{c}.parquet'); assert (S['rowid'].to_numpy()==F['rowid']).all()
    cols=[x for x in S.columns if x!='rowid']
    G={'noise':None,'cl':[x for x in cols if x.startswith('cl_')],'lc':[x for x in cols if x.startswith('lc_')],'sq':[x for x in cols if x.startswith('sq_')],'rp':[x for x in cols if x.startswith('rp_')],'all':cols,'tm':[x for x in cols if x.startswith('tm_')],'tmp':[x for x in cols if x.startswith('tm_prod')],'tmi':[x for x in cols if x.startswith('tm_item')],'tmbil':[x for x in cols if x.startswith('tm_bil') or x=='tm_cos'],'tmprod':[x for x in cols if x.startswith('tm_prod') or x.startswith('tm_item')]}
    for g in groups:
        if g=='noise': X=np.repeat(np.random.RandomState(c).randn(len(F['y'])//10,1).astype(np.float32),10,1).reshape(-1,1)
        else: X=S.select(G[g]).to_numpy().astype(np.float32)
        lib.report(f'v12sig_{g}',F,X,log='../dw_v12sig/results.jsonl')
