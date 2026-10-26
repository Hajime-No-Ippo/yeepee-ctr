import os
V2='/workspace/huawei/v2work'; V3='/workspace/huawei/v3'
FILES={'prep.py':f'{V2}/prep.py','v3fold.py':f'{V2}/v3fold.py','v3feat_ne.py':f'{V2}/v3feat_ne.py','build_folds.py':f'{V2}/build_folds.py',
 'feats.py':f'{V2}/feats.py','extra.py':f'{V2}/extra.py','extra2.py':f'{V2}/extra2.py','cl.py':f'{V2}/cl.py','ens.py':f'{V2}/ens.py',
 'write_sub.py':f'{V2}/write_sub.py','validate.py':f'{V2}/validate.py','blend_lite.py':f'{V2}/lite_build/blend_lite.py','lrfeat2.py':f'{V2}/lrfeat2.py','ttfeat.py':f'{V2}/ttfeat.py',
 'v3/prep.py':f'{V3}/prep.py','v3/features.py':f'{V3}/features.py','v3/split.py':f'{V3}/split.py'}
SRC={}
for k,p in FILES.items():
    s=open(p).read()
    s=s.replace(V2,'@@WORK@@').replace(V3,'@@WORK@@/v3').replace('/workspace/huawei/data','@@RAW@@')
    if k=='build_folds.py':
        s=s.replace("for C,W,name in specs:","""if len(sys.argv)>1 and sys.argv[1]=='lite':   # v4_lite: only what the final fit needs (h1..h5 + val26 = f1..f6, eval)
    specs=[(J20+k*DAY,DAY,f'h{k}') for k in (1,2,3,4,5)]+[(J20+6*DAY,DAY,'val26'),('eval',None,'eval')]
for C,W,name in specs:""")
        assert "=='lite'" in s
    if k=='cl.py':
        a="os.path.getmtime(f'{P3}/{n}.parquet')]"; assert a in s
        s=s.replace(a,"[os.path.getmtime(f) for f in (f'{P3}/{n}.parquet',f'{P2}/{V2_OF[n]}.parquet',f'{D}/feat_x/{n}.parquet',f'{D}/feat_x2/{n}.parquet')]+[os.path.getmtime(f'{D}/{x}/{n}.parquet') for x in extra]]")
        s=s.replace("        import mods; a=mods.apply(a,n,variant)\n","        raise ValueError('mods not used in v7')\n")
    assert '/workspace/' not in s, (k,[l for l in s.splitlines() if '/workspace/' in l])
    SRC[k]=s
head=open(f'{V2}/lite_build/driver_head_v7.py').read()
body=open(f'{V2}/lite_build/driver_body_v7.py').read()
open(f'{V2}/out/solution_v7.py','w').write(head+'\nSRC = '+repr(SRC)+'\n\n'+body)
print('ok',len(SRC))
