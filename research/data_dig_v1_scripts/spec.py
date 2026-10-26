exec(open('svdvar.py').read().split('D=pl.read_parquet')[0])
v=np.ones(cnt.height,np.float32)
M=sp.csr_matrix((v,(cnt['u'].to_numpy(),cnt['ix'].to_numpy())),shape=(cnt['u'].max()+1,NI))
pop=np.asarray(M.sum(0)).ravel(); du=np.asarray(M.sum(1)).ravel()
for a in (0.5,1.0):
    Mn=M.multiply(1/(pop+5)**a).tocsr()
    _,s,_=svds(Mn,k=40,random_state=0); print('alpha',a,np.round(np.sort(s)[::-1],3))
# symmetric normalization D_u^-1/2 M D_i^-1/2
Mn=sp.diags(1/np.sqrt(du+1))@M@sp.diags(1/np.sqrt(pop+1))
_,s,_=svds(Mn.tocsr(),k=40,random_state=0); print('sym',np.round(np.sort(s)[::-1],3))
