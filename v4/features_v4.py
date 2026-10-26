"""v4 features = v3 build() (unchanged, imported) + extra groups:
  pd_   *_per_day: count columns / days in fit window (C - Jan20)
  fi_   per-user taste vector fold-in on fixed k64 SVD item vectors (ridge over P_rows impressions + click log)
  es_   event-state features for the event model: user's recent-event click streak, events in last 24h,
        event-level max/mean/std of con05 / hcos / fold-in, number of candidates already in history
All label-derived inputs come from P_rows (ts < C), asserted."""
import time
import numpy as np, polars as pl
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
import common  # noqa: sets sys.path to v3
from features import build as build_v3, click_log, log
from split import J20, DAY

PD_COLS = ['it_imp', 'it_clk', 'it_nclk_all', 'it_nusers', 'it_nclk_warm', 'us_nev', 'us_nclk_win', 'us_nclk_all',
           'us_nclk_warm', 'us_nuniq', 'hist_imp', 'hist_noclk', 'hist_n_all', 'hist_n_win', 'hist_n_warm']
FI_GRID = [(10, 2), (1, 2)]   # full {1,10}x{0,2} grid + lam 30..300 checked on val25 single-feature (logs/fi_grid.log); best two kept
TRI = None

def _svd_vn(CL, NU, NI, k=64):
    """Same construction as v3 sv block (tf-idf, row-normalised user x item, randomized_svd seed 0) -> normalised item vectors."""
    cnt = CL.group_by('u', 'ii').agg(pl.len().alias('n'))
    Xc = sparse.csr_matrix((np.ones(cnt.height, np.float32), (cnt['u'].to_numpy(), cnt['ii'].to_numpy())), shape=(NU, NI))
    df_ = np.asarray(Xc.sum(0)).ravel(); idf = np.log((NU + 1) / (df_ + 1)).astype(np.float32)
    Xc = Xc @ sparse.diags(idf); rn = np.sqrt(np.asarray(Xc.multiply(Xc).sum(1)).ravel()) + 1e-6; Xc = sparse.diags(1 / rn) @ Xc
    U, S, Vt = randomized_svd(Xc, k, random_state=0, n_iter=5)
    V = (Vt.T * np.sqrt(S)).astype(np.float32)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9), cnt

def fold_in(P_rows, cnt, Vn, NU, NI, grid=FI_GRID, chunk=5000):
    """w_u = (V^T W V + lam I)^-1 V^T W y ; impressions (y=clicked, w=1) + click log (y=1, w=1+a*count)."""
    k = Vn.shape[1]; iu = np.triu_indices(k)
    outer = (Vn[:, iu[0]] * Vn[:, iu[1]]).astype(np.float32)          # NI x k(k+1)/2
    imp = P_rows.group_by('u', 'ii').agg(pl.len().alias('n'), pl.col('clicked').sum().alias('c'))
    def sp(df, w):
        return sparse.csr_matrix((w.astype(np.float32), (df['u'].to_numpy(), df['ii'].to_numpy())), shape=(NU, NI))
    M_imp = sp(imp, imp['n'].to_numpy()); Y_imp = sp(imp, imp['c'].to_numpy())
    M_c1 = sp(cnt, np.ones(cnt.height)); M_cn = sp(cnt, cnt['n'].to_numpy())
    M0 = (M_imp + M_c1).tocsr(); Y0 = (Y_imp + M_c1).tocsr()
    has = (np.asarray(M_imp.sum(1)).ravel() + np.asarray(M_c1.sum(1)).ravel()) > 0
    W = {g: np.zeros((NU, k), np.float32) for g in grid}
    for s in range(0, NU, chunk):
        e = min(NU, s + chunk)
        G0 = M0[s:e] @ outer; Gn = M_cn[s:e] @ outer
        b0 = Y0[s:e] @ Vn; bn = M_cn[s:e] @ Vn
        for (lam, a) in grid:
            Gt = G0 + a * Gn
            G = np.zeros((e - s, k, k), np.float64); G[:, iu[0], iu[1]] = Gt; G[:, iu[1], iu[0]] = Gt
            G[:, np.arange(k), np.arange(k)] += lam
            b = (b0 + a * bn).astype(np.float64)
            W[(lam, a)][s:e] = np.linalg.solve(G, b[..., None])[..., 0]
    return W, has

def build(C, T_ev, T_rows, TE, TR, NI, NU):
    t0 = time.time()
    P_ev = TE.filter(pl.col('ts') < C)
    P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    assert P_rows['ts'].max() < C and T_ev['ts'].min() >= C
    R = build_v3(C, T_ev, T_rows, TE, TR, NI, NU)
    ndays = (C - J20) / DAY
    # ---- pd_: per-day versions of count columns
    R = R.with_columns([(pl.col(c).cast(pl.Float32) / ndays).alias(f'pd_{c}') for c in PD_COLS if c in R.columns])
    # ---- fi_: fold-in taste vectors
    CL = click_log(C, P_ev, P_rows, T_ev)
    assert CL['ts'].max() < C
    Vn, cnt = _svd_vn(CL, NU, NI)
    W, has = fold_in(P_rows, cnt, Vn, NU, NI)
    uu = R['u'].to_numpy().astype(np.int64); cand = Vn[R['ii'].to_numpy().astype(np.int64)]
    fi = {f'fi_l{l}_a{a}_dot': (W[(l, a)][uu] * cand).sum(1) for (l, a) in FI_GRID}
    fi['fi_cold'] = (~has[uu]).astype(np.float32)
    R = R.with_columns([pl.Series(c, v.astype(np.float32)) for c, v in fi.items()])
    R = R.with_columns([((pl.col(c) - pl.col(c).mean().over('eid')) / (pl.col(c).std().over('eid') + 1e-6)).alias(f'fi_{c[3:]}_z')
                        for c in fi if c != 'fi_cold'])
    # ---- es_: event-state (user streak before C) + event-level summaries
    evc = (P_rows.group_by('eid').agg(pl.col('u').first(), pl.col('ts').first(), pl.col('clicked').max().alias('anyc'))
           .sort('u', 'ts'))
    st = evc.group_by('u').agg(pl.col('anyc').tail(1).mean().alias('es_clk_last1'), pl.col('anyc').tail(3).mean().alias('es_clk_last3'),
                               pl.col('anyc').tail(5).mean().alias('es_clk_last5'), pl.col('anyc').mean().alias('es_ev_ctr'),
                               (pl.col('ts') >= C - DAY).sum().alias('es_nev24'), (pl.col('ts') >= C - 6 * 3600).sum().alias('es_nev6'),
                               pl.col('ts').filter(pl.col('anyc') == 1).max().alias('_lastclkev'))
    R = R.join(st, on='u', how='left').with_columns(
        (pl.col('ts') - pl.col('_lastclkev')).fill_null(30 * DAY).alias('es_since_clkev'),
        pl.col('es_nev24').fill_null(0), pl.col('es_nev6').fill_null(0)).drop('_lastclkev')
    ex = []
    for c in ('sv_k64_con05', 'sv_k32_hcos', 'fi_l10_a2_dot'):
        x = pl.col(c).cast(pl.Float64)
        ex += [x.max().over('eid').alias(f'es_{c}_max'), x.mean().over('eid').alias(f'es_{c}_mean'), x.std().over('eid').alias(f'es_{c}_std')]
    ex += [pl.col('hist_in').cast(pl.Int32).sum().over('eid').alias('es_nin_hist')]
    R = R.with_columns(ex)
    log(f'v4 build C={C} rows={R.height} cols={R.width} {time.time()-t0:.0f}s')
    return R
