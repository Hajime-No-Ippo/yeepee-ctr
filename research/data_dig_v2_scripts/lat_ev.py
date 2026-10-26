from sklearn.metrics import roc_auc_score
from h import wauc
_D=pl.read_parquet('/tmp/dig/D.parquet',columns=['eid','clicked','cut'])
def q(x,nm):
    dd=_D.with_columns(pl.Series('x',np.asarray(x,float)))
    g={c:round(roc_auc_score(dd.filter(pl.col('cut')==c)['clicked'],dd.filter(pl.col('cut')==c)['x']),4) for c in (25,26)}
    w={c:round(wauc(dd.filter(pl.col('cut')==c),'x'),4) for c in (25,26)}
    print(nm,'g',g,'w',w,flush=True)
