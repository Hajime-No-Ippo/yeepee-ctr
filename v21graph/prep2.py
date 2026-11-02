import sys,os; sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/huawei/dw_signals'); from common import *
D=Data(); y=D.Y; has=y.sum(1)>0; pos=np.where(has,y.argmax(1),-1)
np.savez('/workspace/huawei/v21graph/out/slates.npz',e=np.arange(D.NE),u=D.EU,cand=D.RI,pos=pos.astype(np.int8))
print(D.NE,has.mean())
