import sys,json; sys.path.insert(0,'/workspace/huawei/dw_p33')
from p33lib import *
from scipy.sparse.linalg import svds
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=Data(); X=Jd(25); S=D.snap(X); u,i=S['cu'],S['ci']
key=np.unique(u.astype(np.int64)*D.NI+i); uu=(key//D.NI).astype(np.int32); ii=(key%D.NI).astype(np.int32)
nu=np.bincount(uu,minlength=D.NU); ni=np.bincount(ii,minlength=D.NI); print('click pairs',len(key),'users',(nu>0).sum(),'items',(ni>0).sum())
def spec(uu,ii,k=64):
    n_=np.bincount(ii,minlength=D.NI); M=sparse.csr_matrix((1/np.sqrt(n_[ii]+5),(uu,ii)),shape=(D.NU,D.NI),dtype=np.float32)
    s=svds(M,k=k,random_state=0,return_singular_vectors=False); return np.sort(s)[::-1]
s_real=spec(uu,ii)
# null: shuffle item ids within column-popularity-preserving (permute the items across the whole log keeping user degree & item pop)
rs=np.random.RandomState(0); ii_n=rs.permutation(ii); k2=np.unique(uu.astype(np.int64)*D.NI+ii_n); s_null=spec((k2//D.NI).astype(np.int32),(k2%D.NI).astype(np.int32))
print('click-matrix sv real',np.round(s_real[:24],2)); print('null',np.round(s_null[:24],2))
# impression matrix too
nX=int(np.searchsorted(D.ETS,X)); ui=np.repeat(D.EU[:nX],10); iti=D.RI[:nX].ravel()
Mi=sparse.csr_matrix((np.ones(len(ui),np.float32),(ui,iti)),shape=(D.NU,D.NI)); Mi.sum_duplicates()
nu_=np.asarray(Mi.sum(1)).ravel(); ni_=np.asarray(Mi.sum(0)).ravel(); Mc=Mi.tocoo(); Ms=sparse.csr_matrix((Mc.data/np.sqrt((nu_[Mc.row]+1)*(ni_[Mc.col]+1)),(Mc.row,Mc.col)),shape=Mi.shape,dtype=np.float32)
s_imp=np.sort(svds(Ms,k=64,random_state=0,return_singular_vectors=False))[::-1]; print('impression sv',np.round(s_imp[:24],2))
sp=np.load('sv_ppmi_W20inv.npy'); sp5=np.load('sv_ppmi_W5inv.npy'); sp60=np.load('sv_ppmi_W60one.npy')
np.savez('spectra.npz',click=s_real,click_null=s_null,imp=s_imp,ppmi20=sp,ppmi5=sp5,ppmi60=sp60)
fig,ax=plt.subplots(1,3,figsize=(16,4.2))
ax[0].plot(s_real,'o-',ms=3,label='click matrix (train<C25)'); ax[0].plot(s_null,'x-',ms=3,label='null: item ids permuted'); ax[0].set_title('user x item click matrix sv'); ax[0].legend(); ax[0].set_yscale('log')
ax[1].plot(s_imp,'o-',ms=3); ax[1].set_title('user x item IMPRESSION matrix sv'); ax[1].set_yscale('log')
for s,l in ((sp5,'W5 inv'),(sp,'W20 inv'),(sp60,'W60 bag')): ax[2].plot(s/s[1],label=l)
ax[2].set_ylim(0,1.0); ax[2].set_title('PPMI-SVD sv (rel. to sv[1])'); ax[2].legend(); ax[2].axvline(12,ls=':',c='k')
plt.tight_layout(); plt.savefig('spectra.png',dpi=110)
