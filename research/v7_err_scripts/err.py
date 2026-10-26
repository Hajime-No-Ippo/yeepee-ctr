# v7 error analysis on val25/val26 from saved ens/ predictions (no retrain). Eval shares use eval features only (no labels).
import sys, json, numpy as np, polars as pl
sys.path.insert(0,'/workspace/huawei/v2work')
from cl import matrix
from sklearn.metrics import roc_auc_score as A
EX=('feat_lr2','feat_tt'); DAY=86400
L=lambda t,c,s,ci: np.load(f'/workspace/huawei/v2work/ens/{t}_c{c}_s{s}.npy')[:,ci]
NEED=['us_since_ev','us_since_clk','us_hlen','us_nclk_all','it_imp']
def frame(n):
    X,m,f=matrix(n,1.0,extra=EX)
    d=m.with_columns([pl.Series(c,X[:,f.index(c)]) for c in NEED])
    la=np.fmin(d['us_since_ev'].to_numpy(),d['us_since_clk'].to_numpy())
    dow=((d['ts'].to_numpy()//DAY)+3)%7   # 0=Mon (epoch day 0 = Thu)
    seg=lambda x,b,l: np.array(l,dtype=object)[np.digitize(x,b)]
    d=d.with_columns(
      pl.Series('user', np.where(d['us_nohist'].to_numpy()==1,'empty',np.where(d['us_cold'].to_numpy()==1,'history-only','warm'))),
      pl.Series('lastact', seg(la,[3600,DAY],['<1h','1-24h','>=1d'])),
      pl.Series('stale', seg(d['tm_stale'].to_numpy(),[DAY,2*DAY],['<1d','1-2d','>=2d'])),
      pl.Series('dow', np.array(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])[dow]),
    )
    return d
ev=frame('eval')
# decile edges from eval-independent train-ish reference: use val25 edges (labels not used)
out={}; rep=[]
def dec(x,edges): return np.digitize(x,edges)
for c in (25,26):
    d=frame(f'val{c}')
    b0,b1,sl=L('v7base',c,0,1),L('v7base',c,1,1),L('v7slow',c,0,2); p=((b0+b1)/2+sl)/2
    d=d.with_columns(pl.Series('p',p)); y=d['clicked'].to_numpy(); tot=A(y,p)
    if c==25:
        ue=np.unique(np.nanpercentile(d.group_by('eid').agg(pl.col('us_nclk_all').first())['us_nclk_all'].to_numpy(),np.arange(10,100,10)))
        ie=np.unique(np.nanpercentile(d['it_imp'].to_numpy(),np.arange(10,100,10)))
    for D in (d,):
        pass
    d=d.with_columns(pl.Series('act_dec',dec(d['us_nclk_all'].to_numpy(),ue)),pl.Series('item_dec',dec(d['it_imp'].to_numpy(),ie)))
    E=d.group_by('eid').agg(pl.col('clicked').max().alias('ey'))
    d=d.join(E,on='eid',how='left'); d=d.with_columns(pl.when(pl.col('ey')==1).then(pl.lit('has-click')).otherwise(pl.lit('no-click')).alias('evclick'))
    if c==25: ev=ev.with_columns(pl.Series('act_dec',dec(ev['us_nclk_all'].to_numpy(),ue)),pl.Series('item_dec',dec(ev['it_imp'].to_numpy(),ie)))
    res={'auc':round(tot,5),'rows':d.height,'ctr':round(float(y.mean()),5),'segs':{}}
    for s in ('user','lastact','stale','dow','act_dec','item_dec','evclick'):
        R=[]
        for k in sorted(d[s].unique().to_list()):
            q=d.filter(pl.col(s)==k); yy=q['clicked'].to_numpy()
            a=round(A(yy,q['p'].to_numpy()),5) if 0<yy.sum()<len(yy) else None
            es=float((ev[s]==k).mean()) if s in ev.columns else None
            R.append(dict(seg=str(k),share=round(q.height/d.height,4),ctr=round(float(yy.mean()),5),auc=a,eval_share=None if es is None else round(es,4)))
        res['segs'][s]=R
    # between / within decomposition
    g=d.group_by('eid').agg(pl.col('ey').first(),(1-(1-pl.col('p').map_batches(lambda x: 1/(1+np.exp(-x)))).product()).alias('pe'))
    rk=d.filter(pl.col('ey')==1).with_columns(pl.col('p').rank('average').over('eid').alias('rk')).filter(pl.col('clicked')==1)['rk'].to_numpy()
    j=d.join(g.select('eid','pe'),on='eid',how='left'); pe=j['pe'].to_numpy(); ey=j['ey'].to_numpy()
    # oracle within: keep event score, clicked row ranked first inside its event
    pw=np.log(pe+1e-12)+y*0+0; ow=A(y, pe + y*1e-6 + (ey==1)*(0) ) if False else None
    zr=d.with_columns(pl.col('p').rank('ordinal').over('eid').alias('r'))['r'].to_numpy()
    # within-oracle: same event-level score pe, within-event order: clicked=top, others by model
    o_within=A(y, np.log(pe+1e-12)*1000 + np.where(y==1,20,zr))
    # between-oracle: no-click events pushed to the bottom, within-event model scores kept
    o_between=A(y, p + np.where(ey==1,0,-1e3))
    res['decomp']=dict(event_auc=round(A(g['ey'].to_numpy(),g['pe'].to_numpy()),5),
        within_auc=round(float(((rk-1)/9).mean()),5),top1=round(float((rk==10).mean()),4),
        oracle_within=round(o_within,5),oracle_between=round(o_between,5),
        gap_within=round(o_within-tot,5),gap_between=round(o_between-tot,5),event_click_rate=round(float(g['ey'].mean()),4))
    out[c]=res
json.dump(out,open('/workspace/huawei/research/v7_err_scripts/err.json','w'),indent=1)
for c in out:
    print(c,out[c]['auc'],out[c]['decomp'])
    for s,R in out[c]['segs'].items(): print(' ',s,[(r['seg'],r['share'],r['eval_share'],r['ctr'],r['auc']) for r in R])
