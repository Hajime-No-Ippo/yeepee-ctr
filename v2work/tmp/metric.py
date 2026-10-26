"""Global ROC AUC over all rows (the competition metric; not per-event averaged)."""
import numpy as np
from sklearn.metrics import roc_auc_score

def global_auc(y, p):
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    if y.min() == y.max():
        return float('nan')
    return float(roc_auc_score(y, p))
