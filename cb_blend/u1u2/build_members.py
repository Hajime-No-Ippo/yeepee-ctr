"""Per-seed graph feature cache + rowid / slate-structure verification. Read-only on v21graph."""
import sys,glob,os,json,time; sys.dont_write_bytecode=True; sys.path.insert(0,'/workspace/huawei/cb_blend/u1u2')
import numpy as np, gfeat_chunk as gc
H='/workspace/huawei/cb_blend'; D=H+'/u1u2'
CFG={23:('u1b',3,'800'),25:('u1',5,'350'),26:('u1',6,'350')}   # as in v21graph/ms.py
ALT={23:('u1',3,'350')}                                            # C23 with the 350-step u1 config (seed 0 only exists)
log={}
for c,(tag,k,st) in CFG.items():
    T=gc.load_tab(f'val{c}'); o=np.load(f'{H}/oof_c{c}.npz',allow_pickle=True); 
    ok=bool((T['rowid'].astype(str)==o['rowid'].astype(str)).all()) if 'rowid' in o.files else None
    sl=(T['user'].reshape(-1,10)==T['user'].reshape(-1,10)[:,:1]).all() and (T['y'].astype(int)==o['y'].astype(int)).all()
    print(f'C{c}: n={len(T["item"])} rowid==oof rowid {ok}  y equal & slates contiguous-10 same user {bool(sl)}',flush=True); log[c]=dict(rowid_ok=ok,y_slate_ok=bool(sl))
    jobs=[(tag,k,st)]+([ALT[c]] if c in ALT else [])
    for tg,kk,ss in jobs:
        for f in sorted(glob.glob(gc.O+f'emb_{tg}_k{kk}_s*.npz')):
            s=int(f.split('_s')[-1][:-4]); out=f'{D}/feat_c{c}_{tg}_s{s}.npz'
            if os.path.exists(out): continue
            try: e=np.load(f); U,V=e['U'+ss],e['V'+ss]
            except Exception as ex: print('skip',f,ex); continue
            t=time.time(); F=gc.feats_chunk(T,U,V); np.savez(out,**F); print(f'  {os.path.basename(out)} {time.time()-t:.0f}s',flush=True)
json.dump(log,open(D+'/align.json','w'))
