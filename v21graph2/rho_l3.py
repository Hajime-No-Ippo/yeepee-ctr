from evalrho import *; from ana import *
print('G7 (old 7s L3 sum3) rho EN %.4f'%rho(G7))
A=avg('u1',27,SEEDORD[:7],350); Z=np.stack([zs(A[:,j]) for j in range(4)],1)
for nm,w in (('sum3',(1,1,1,0)),('sum3b',(2,0,1,0)),('s_uv',(1,0,0,0)),('s_h',(0,1,0,0)),('cos_mean',(0,0,1,0))):
    print(nm,'rho EN %.4f'%rho(to_merged(Z@np.array(w,float))))
