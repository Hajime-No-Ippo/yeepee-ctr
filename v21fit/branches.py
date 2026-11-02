exec(open('reopt2.py').read().split("out={}")[0])
from scipy.stats import norm
import pickle
out=pickle.load(open('reopt_out.pkl','rb')); p0=out[1]
pS1=np.load('pS1.npy'); pS2=np.load('pS2.npy'); q5=pR+2*(pS2-pR)
cands={'S1':pS1,'S2':pS2,
 'a1 (2/3 full step)':pR+.667*(p0-pR),'a2 (full step)':p0,
 'b (v5 held, 3/4 step)':pR+.75*(q5-pR),'b2 (v5 held, full)':q5,
 'c (R1, q=.75)':np.array([.136,.098,.75,-.03,-.015]),'c2 (R1, q=.6)':np.array([.136,.098,.6,-.03,-.015])}
d,R=dE,RE
gR={k:A_(mk(p),d,R)-A_(wR,d,R) for k,p in cands.items()}
def line(k,ref,Wt=None,lab=''):
    g=A_(mk(cands[k]),d,R)-A_(mk(cands[ref]) if ref!='R1' else wR,d,R)
    if Wt is None: Wt=np.ones(len(g))/len(g)
    o=np.argsort(g); c=np.cumsum(Wt[o]); q=lambda x:g[o][np.searchsorted(c,x)]
    print('%-24s vs %-3s %s mean %+.6f 5/95 %+.6f %+.6f P>0 %.2f'%(k,ref,lab,(Wt*g).sum(),q(.05),q(.95),(Wt*(g>0)).sum()))
print('params:'); [print(k,np.round(p,3)) for k,p in cands.items()]
print('--- unconditional (posterior given R1,Q2)')
for k in cands: line(k,'R1')
# scenarios: S1,S2 outcome vs R1 with obs noise sd .00003 on the difference
s1=gR['S1'];s2=gR['S2']; z=.00003
P1=norm.cdf(s1/z);P2=norm.cdf(s2/z)
for nm,Wt in (('both beat R1',P1*P2),('only S2 beats',(1-P1)*P2),('only S1 beats',P1*(1-P2)),('both lose',(1-P1)*(1-P2))):
    print('scenario',nm,'prob %.2f'%Wt.mean()); Wt=Wt/Wt.sum()
    for k in cands:
        if k in('S1','S2'): continue
        line(k,'R1',Wt,'|sc'); 
    if nm=='both beat R1':
        for k in('a1 (2/3 full step)','a2 (full step)'): line(k,'S1',Wt,'|sc'); line(k,'S2',Wt,'|sc')
    if nm=='only S2 beats':
        for k in('b (v5 held, 3/4 step)','b2 (v5 held, full)'): line(k,'S2',Wt,'|sc')
