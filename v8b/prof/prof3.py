import sys, time, cProfile, pstats
sys.path.insert(0,'/workspace/huawei/v8b/src'); sys.path.insert(1,'/workspace/huawei/v3')
import importlib.util
src=open('/workspace/huawei/v8b/src/v3feat_ne.py').read().replace('@@WORK@@','/workspace/huawei/v8')
open('v3feat_ne_p.py','w').write(src)
sys.path.insert(0,'.')
from v3feat_ne_p import load, build
from split import J20, DAY, first_events
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
pr=cProfile.Profile(); t=time.time(); pr.enable()
R = build(J20 + 7 * DAY, EE, ER, TE, TR, NI, NU)
pr.disable(); print('build', time.time()-t)
pstats.Stats(pr).sort_stats('tottime').print_stats(15)
