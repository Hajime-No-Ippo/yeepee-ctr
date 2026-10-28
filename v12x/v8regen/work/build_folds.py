import polars as pl, sys, os
sys.path.insert(0,'/workspace/huawei/v12x/v8regen/work')
from feats import load, first_events, build, log
D='/workspace/huawei/v12x/v8regen/work'; os.makedirs(f'{D}/feat',exist_ok=True)
TE,TR,EE,ER,NI=load(D)
DAY=86400; J20=1737331200
END=J20+7*DAY; VC=J20+4*DAY
# validation: V = first event/user in [Jan24,Jan27); training folds end before Jan24
specs=[(VC,END-VC,'V')]+[(J20+k*DAY,VC-(J20+k*DAY),f'g{k}') for k in (1,2,3)]
if len(sys.argv)>1 and sys.argv[1]=='c25':
    specs=[(J20+5*DAY,2*DAY,'val25')]+[(J20+k*DAY,DAY,f'h{k}') for k in (1,2,3,4)]
if len(sys.argv)>1 and sys.argv[1]=='c26':
    specs=[(J20+6*DAY,DAY,'val26')]+[(J20+k*DAY,DAY,f'h{k}') for k in (5,)]+[('eval',None,'eval')]
if len(sys.argv)>1 and sys.argv[1]=='eft':
    specs=[(J20+k*DAY,(J20+5*DAY)-(J20+k*DAY),f'e{k}') for k in (1,2,3,4)]+[(J20+k*DAY,END-(J20+k*DAY),f'E{k}') for k in (1,2,3,4,5,6)]
if len(sys.argv)>1 and sys.argv[1]=='dft':
    specs=[(J20+k*DAY,(J20+6*DAY)-(J20+k*DAY),f'd{k}') for k in (1,2,3,4,5)]
if len(sys.argv)>1 and sys.argv[1]=='final':
    specs=[(J20+k*DAY,END-(J20+k*DAY),f'F{k}') for k in (1,2,3,4,5,6)]+[('eval',None,'eval')]
if len(sys.argv)>1 and sys.argv[1]=='lite':   # v4_lite: only what the final fit needs (h1..h5 + val26 = f1..f6, eval)
    specs=[(J20+k*DAY,DAY,f'h{k}') for k in (3,4,5)]+[(J20+6*DAY,DAY,'val26'),('eval',None,'eval')]
for C,W,name in specs:
    out=f'{D}/feat/{name}.parquet'
    if os.path.exists(out): continue
    if C=='eval':
        C=J20+7*DAY; T_ev=EE; T_rows=ER
    else:
        T_ev=first_events(TE,C,W); T_rows=TR.join(T_ev.select('eid'),on='eid')
    R=build(C,T_ev,T_rows,TE,TR,NI)
    R.drop('h','h_ts').write_parquet(out)
    log('wrote',name,R.shape)
