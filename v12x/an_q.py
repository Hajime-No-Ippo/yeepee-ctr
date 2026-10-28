from analyze import *
import glob
W=[0,.25,.5,.75,1,1.5,2,3]
for tag in ('lrank','xendcg'):
    for ci,rn in ((0,500),(1,1000)):
        res={}
        for c in (25,23):
            p=f'{X}/{tag}_c{c}_s0.npy'
            if not os.path.exists(p): continue
            big,sm,ml=base(c); z0=(big+sm+2*ml)/4; r=ld(p,ci); rs=r/r.std()*big.std()   # scale to dwbig logit sd
            rd=demean(rs); res[c]=dict(alone=auc(c,r),
              member={w:auc(c,(big+sm+2*ml+w*rs)/(4+w)) for w in (.5,1,2)},
              big_plus_demeaned={w:auc(c,big+w*rd) for w in W}, v12_plus_demeaned={w:auc(c,z0+w*rd) for w in W}, v12=auc(c,z0), dwbig=auc(c,big))
        if res: print(tag,rn,json.dumps(res))
for c in (25,23):
    for f in sorted(glob.glob(f'{X}/*_c{c}_s0.npy')):
        t=os.path.basename(f)[:-len(f'_c{c}_s0.npy')]
        if t.startswith(('mb','f_','lgb255')):
            a=np.load(f); print(c,t,[auc(c,a[:,i]) for i in range(a.shape[1])])
