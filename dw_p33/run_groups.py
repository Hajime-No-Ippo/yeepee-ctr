import sys; sys.path.insert(0,'/workspace/huawei/dw_lb'); sys.dont_write_bytecode=True
from lib import *
import lib
cuts=[int(x) for x in sys.argv[1].split(',')]; groups=sys.argv[2].split(',')
for c in cuts:
    F=load_cut(c); S=pl.read_parquet(f'/workspace/huawei/dw_p33/cache/pp_c{c}.parquet'); assert (S['rowid'].to_numpy()==F['rowid']).all()
    cols=[x for x in S.columns if x!='rowid']
    sw=lambda *p:[x for x in cols if x.startswith(p)]
    G={'cos':[x for x in cols if x.startswith(('pp12_cos','pp8','pp16','pp12x','ppb12','pp12_unorm','pp12_ucos'))],
       'cos1':['pp12_cos','pp12_cos_z','pp12_cos_dm'],
       'rec':sw('pp12_rec','pp12_last3','pp12_max5','pp12_decay5'),'prod':sw('pp_prod'),'item':sw('pp_item'),'user':sw('pp_user'),'multi':sw('pp12_w3'),
       'prodit':sw('pp_prod','pp_item'),'all':cols}
    G['cosprod']=G['cos']+G['prod']
    for g in groups:
        if g=='noise': X=np.repeat(np.random.RandomState(c).randn(len(F['y'])//10,1).astype(np.float32),10,1).reshape(-1,1)
        else: X=S.select(G[g]).to_numpy().astype(np.float32)
        lib.report(f'p33_{g}',F,X,log='../dw_p33/results.jsonl')
