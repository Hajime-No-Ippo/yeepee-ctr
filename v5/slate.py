"""Step B: slate-context features from each event's own 10 candidates (item factors / popularity from CLg = train-only clicks, ts < C).
Usage: python slate.py precheck | NAME...  -> v5/feat_sl/NAME.parquet
 sl_cand_cent_cos: candidate's mean cosine to the other 9; sl_cand_top_cos: max cosine to the other 9
 sl_disp: 1 - mean pairwise cosine of the slate (constant within event); sl_user_cent: user click centroid vs slate centroid (constant within event)
 sl_pop_rank: rank of train-only click popularity within the slate (1 = least popular)"""
import sys, os, json, time, numpy as np, polars as pl
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/workspace/huawei/v3')
from features import load, click_log
from split import J20, DAY, first_events
D = '/workspace/huawei/v5'; OUT = f'{D}/feat_sl'; os.makedirs(OUT, exist_ok=True)
TE, TR, EE, ER, NI, NU = load('/workspace/huawei/v3/data')
SPEC = {**{f'f{k}': (J20 + k * DAY, DAY) for k in range(1, 7)}, 'val23': (J20 + 3 * DAY, None), 'val25': (J20 + 5 * DAY, None), 'val26': (J20 + 6 * DAY, None), 'eval': (J20 + 7 * DAY, 'eval')}
def nrm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
def csr(u, i, w): return sparse.csr_matrix((np.asarray(w, np.float32), (u, i)), shape=(NU, NI))
def build(C, T_ev, T_rows, only_cent=False):
    P_ev = TE.filter(pl.col('ts') < C); P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    CL = click_log(C, P_ev, P_rows, T_ev); CLg = click_log(C, P_ev, P_rows, T_ev.head(0))
    assert CL['ts'].max() < C and CLg['ts'].max() < C and P_rows['ts'].max() < C
    R = T_rows.join(T_ev.select('eid', 'u'), on='eid', how='inner').select('rowid', 'eid', 'u', 'ii', *(['clicked'] if 'clicked' in T_rows.columns else [])).sort('eid', 'rowid')
    n_per = R.group_by('eid').agg(pl.len().alias('n'))['n']; assert n_per.min() == 10 and n_per.max() == 10, 'events must have exactly 10 rows'
    ii = R['ii'].to_numpy().astype(np.int64); uu = R['u'].to_numpy().astype(np.int64); nE = R.height // 10
    cg = CLg.group_by('u', 'ii').agg(pl.len().alias('n'))
    Xc = csr(cg['u'].to_numpy(), cg['ii'].to_numpy(), np.ones(cg.height)); df_ = np.asarray(Xc.sum(0)).ravel(); idf = np.log((NU + 1) / (df_ + 1)).astype(np.float32)
    Xc = Xc @ sparse.diags(idf); Xc = sparse.diags(1 / (np.sqrt(np.asarray(Xc.multiply(Xc).sum(1)).ravel()) + 1e-6)) @ Xc
    U, S, Vt = randomized_svd(Xc, 32, random_state=0, n_iter=5); Vn = nrm(Vt.T * np.sqrt(S)).astype(np.float32)
    V = Vn[ii].reshape(nE, 10, -1); G = np.einsum('eik,ejk->eij', V, V); idx = np.arange(10); G[:, idx, idx] = np.nan
    F = {'sl_cand_cent_cos': np.nanmean(G, 2).ravel()}
    if not only_cent:
        F['sl_cand_top_cos'] = np.nanmax(G, 2).ravel()
        F['sl_disp'] = np.repeat(1 - np.nanmean(G, (1, 2)), 10)
        cnt = CL.group_by('u', 'ii').agg(pl.len()); Sc = csr(cnt['u'].to_numpy(), cnt['ii'].to_numpy(), np.ones(cnt.height))
        nC = nrm(np.asarray(Sc @ Vn)); cent = nrm(V.mean(1))
        F['sl_user_cent'] = np.repeat((nC[uu.reshape(nE, 10)[:, 0]] * cent).sum(1), 10)
        pop = np.bincount(CLg['ii'].to_numpy(), minlength=NI).astype(np.float32)
        F['sl_pop_rank'] = pl.DataFrame({'e': R['eid'], 'p': pop[ii]}).with_columns(pl.col('p').rank('average').over('e'))['p'].to_numpy()
    out = R.with_columns([pl.Series(c, v.astype(np.float32)) for c, v in F.items()])
    if not only_cent:   # acceptance: sl_disp / sl_user_cent constant within event
        chk = out.group_by('eid').agg(pl.col('sl_disp').n_unique().alias('a'), pl.col('sl_user_cent').n_unique().alias('b'))
        assert chk['a'].max() == 1 and chk['b'].max() == 1
    return out
def within(R, col):
    a = []
    for yy, ss in R.group_by('eid').agg('clicked', col).select('clicked', col).iter_rows():
        yy = np.array(yy)
        if 0 < yy.sum() < len(yy): a.append(roc_auc_score(yy, np.array(ss)))
    return float(np.mean(a))
if sys.argv[1] == 'precheck':
    C, _ = SPEC['val25']; T_ev = first_events(TE, C, None); t0 = time.time()
    R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'), only_cent=True)
    w = within(R, 'sl_cand_cent_cos'); g = roc_auc_score(R['clicked'], R['sl_cand_cent_cos'])
    r = dict(step='B_precheck', feature='sl_cand_cent_cos', cut=25, within=round(w, 5), within_flipped=round(1 - w, 5), global_auc=round(g, 5), runtime_s=round(time.time() - t0))
    print(json.dumps(r)); open(f'{D}/out/results.jsonl', 'a').write(json.dumps(r) + '\n')
else:
    for name in sys.argv[1:]:
        t0 = time.time(); C, W = SPEC[name]
        if W == 'eval': R = build(C, EE, ER)
        else:
            T_ev = first_events(TE, C, W); R = build(C, T_ev, TR.join(T_ev.select('eid'), on='eid', how='semi'))
        R.select('rowid', *[c for c in R.columns if c.startswith('sl_')]).write_parquet(f'{OUT}/{name}.parquet'); print(name, R.shape, f'{time.time()-t0:.0f}s', flush=True)
