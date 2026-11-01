import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
dp=lambda a: np.sqrt(2)*norm.ppf(a)
auc=lambda d: norm.cdf(d/np.sqrt(2))
def comb(a1,a2,rho,w):  # w weight on model 2 (std ranks, equal var)
    d1,d2=dp(a1),dp(a2)
    return auc(((1-w)*d1+w*d2)/np.sqrt((1-w)**2+w**2+2*w*(1-w)*rho))
if __name__=='__main__':
    # check 1: E0 (.824558) + O(v21list B line): O alone? public of raw v21list B unknown; p33 probe .818977 ~ O alone approx .8185
    rho=.9582
    for a2 in (.8185,.8190,.8200):
        print('a2',a2,[ (w,round(comb(.824558,a2,rho,w)-.824558,6)) for w in (.1,.3,.5)])
    print('actual: w.1 +.000405, w.3 +.000586')
    rho=.9564
    print('EN+O',[ (w,round(comb(.828824,.8190,rho,w)-.828824,6)) for w in (.05,.1,.2,.3)],'actual w.1 +.000032')
