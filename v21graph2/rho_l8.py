from evalrho import *; from ana import *
import sys
seeds=[int(s) for s in sys.argv[1].split(',')]
A=avg('u12',27,seeds,650); Z=np.stack([zs(A[:,j]) for j in range(4)],1)
for nm,w in (('sum3',(1,1,1,0)),('sum3b',(2,0,1,0)),('s_uv',(1,0,0,0)),('s_h',(0,1,0,0)),('cos_mean',(0,0,1,0)),('uv+cm',(1,0,1,0)),('uv+h',(1,1,0,0))):
    x=to_merged(Z@np.array(w,float)); print(nm,'rho EN %.4f  rho G7 %.4f'%(rho(x),rho(x,nr(G7))),flush=True)
