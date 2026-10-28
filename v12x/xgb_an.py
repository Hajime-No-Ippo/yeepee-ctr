from analyze import *
for c in (25,23):
    big,sm,ml=base(c); z0=(big+sm+2*ml)/4
    print('cut',c,'v12 blend',auc(c,z0),'dwbig s0',auc(c,big))
    try:
        b1=ld(f'{V}/dwbig_c{c}_s1.npy',1); print(' dwbig s1',auc(c,b1))
    except Exception as e: print(e)
    for s in (0,1):
        a=np.load(f'{X}/xgb_c{c}_s{s}.npy'); print(' xgb s%d shape'%s,a.shape,[auc(c,a[:,i]) for i in range(a.shape[1])])
        for ci in range(a.shape[1]):
            x=a[:,ci]
            res={w:auc(c,(big+sm+2*ml+w*x)/(4+w)) for w in (.5,1,2)}
            print('  col',ci,'blend',res)
