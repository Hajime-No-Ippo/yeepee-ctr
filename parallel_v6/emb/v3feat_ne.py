"""Feature builder (as-of-cutoff, leakage-safe) for TechArena CTR.

build(C, T_ev, T_rows, TE, TR) returns one feature row per target candidate.
Every statistic uses ONLY:
  * train events/rows with ts < C (fit window: impressions + labels), and
  * click histories of events with ts < C, plus the target event's OWN history (strictly before its ts).
Since target events are each user's first event at/after C, their histories contain only clicks < C,
so nothing observed after a target row's own event time is ever joined.

Feature groups (column prefixes, used for ablation):
  hist_  candidate vs the user's click log / own history / past impressions of that user-item pair
  it_    item popularity & CTR (full fit window, last 24h, last 6h, trend)
  co_    item-item co-click (user co-occurrence cosine) and sequential transitions vs user's last 5/20 clicks
  us_    user activity / CTR / recency / cold-start flags
  ev_    within-event rank / z-score / event max & mean of the main features
  tm_    time_bucket, day_type, hour, staleness since cutoff
  sv_    truncated SVD (tf-idf, row-normalised user x item click matrix as of C): user/history/item cosines
  mf_    logistic matrix factorisation score fitted on fit-window impressions
"""
import time
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

J20 = 1737331200
DAY = 86400
G_PRIOR = 0.0629
K_ITEM = 20          # pseudo-impressions for item CTR shrinkage
K_USER = 5           # pseudo-events for user CTR shrinkage
TOPK = 200           # co-click neighbours kept per item
ID_COLS = ['rowid', 'eid', 'u', 'ii', 'ts', 'clicked', 'ord']
_t0 = time.time()

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def load(D):
    TE = pl.read_parquet(f'{D}/ev_train.parquet')
    TR = pl.read_parquet(f'{D}/rows_train.parquet')
    EE = pl.read_parquet(f'{D}/ev_eval.parquet')
    ER = pl.read_parquet(f'{D}/rows_eval.parquet')
    NI = pl.read_parquet(f'{D}/items.parquet').height
    NU = pl.read_parquet(f'{D}/users.parquet').height
    return TE, TR, EE, ER, NI, NU

def _hist_long(E):
    return (E.select('u', 'h', 'h_ts').explode(['h', 'h_ts']).drop_nulls('h')
             .select('u', pl.col('h').cast(pl.Int32).alias('ii'), pl.col('h_ts').cast(pl.Int64).alias('ts')))

def click_log(C, P_ev, P_rows, T_ev):
    """Exact (user,item,ts) click log as of C: fit-window clicks + warm-up clicks seen in pre-C / own histories."""
    win = P_rows.filter(pl.col('clicked') == 1).select('u', pl.col('ii').cast(pl.Int32), 'ts')
    hist = pl.concat([_hist_long(P_ev), _hist_long(T_ev)]).filter(pl.col('ts') < C)
    CL = pl.concat([win, hist]).unique().sort('u', 'ts', 'ii')
    return CL

def cooc_tables(CL, NI):
    """User co-occurrence cosine (top-K per item) + lag-weighted transition probabilities prev->next."""
    us = CL.select('u', 'ii').unique()
    pop = us.group_by('ii').agg(pl.len().alias('pop'))
    pairs = (us.join(us, on='u', suffix='_b').filter(pl.col('ii') != pl.col('ii_b'))
               .group_by('ii', 'ii_b').agg(pl.len().cast(pl.Int32).alias('co')))
    co = (pairs.join(pop, on='ii').join(pop.rename({'ii': 'ii_b', 'pop': 'pop_b'}), on='ii_b')
               .with_columns((pl.col('co') / (pl.col('pop') * pl.col('pop_b')).sqrt()).cast(pl.Float32).alias('cos'))
               .select(pl.col('ii_b').alias('j'), 'ii', 'cos')       # j = history item, ii = candidate
               .sort('cos', descending=True).group_by('j').head(TOPK))
    seq = CL.select('u', 'ii')  # already sorted by u, ts
    parts = []
    for lag in (1, 2, 3):
        parts.append(seq.with_columns(pl.col('ii').shift(lag).over('u').alias('j')).drop_nulls('j')
                        .select(pl.col('j').cast(pl.Int32), 'ii', pl.lit(1.0 / lag).alias('w')))
    trn = pl.concat(parts).filter(pl.col('j') != pl.col('ii')).group_by('j', 'ii').agg(pl.col('w').sum())
    tot = trn.group_by('j').agg(pl.col('w').sum().alias('tot'))
    trn = (trn.join(tot, on='j').with_columns((pl.col('w') / (pl.col('tot') + 5.0)).cast(pl.Float32).alias('trp'))
              .select('j', 'ii', 'trp').sort('trp', descending=True).group_by('j').head(TOPK))
    return co, trn

def fit_mf(P_rows, NU, NI, dim=24, epochs=6, lr=0.05, l2=1e-4, seed=0, batch=200_000):
    """Logistic MF  logit = b + bu + bi + <pu, qi>  fitted with Adagrad on fit-window impressions (numpy)."""
    rng = np.random.default_rng(seed)
    u = P_rows['u'].to_numpy().astype(np.int64); i = P_rows['ii'].to_numpy().astype(np.int64)
    y = P_rows['clicked'].to_numpy().astype(np.float32)
    P = (rng.standard_normal((NU, dim)) * 0.01).astype(np.float32); Q = (rng.standard_normal((NI, dim)) * 0.01).astype(np.float32)
    bu = np.zeros(NU, np.float32); bi = np.zeros(NI, np.float32)
    b = np.float32(np.log(y.mean() / (1 - y.mean())))
    gP = np.full_like(P, 1e-6); gQ = np.full_like(Q, 1e-6); gbu = np.full_like(bu, 1e-6); gbi = np.full_like(bi, 1e-6)
    n = len(y)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]; ub, ib, yb = u[idx], i[idx], y[idx]
            pu, qi = P[ub], Q[ib]
            z = b + bu[ub] + bi[ib] + (pu * qi).sum(1)
            g = (1 / (1 + np.exp(-z)) - yb).astype(np.float32)          # dL/dz
            m = len(idx)
            Su = sparse.csr_matrix((np.ones(m, np.float32), (ub, np.arange(m))), shape=(NU, m))
            Si = sparse.csr_matrix((np.ones(m, np.float32), (ib, np.arange(m))), shape=(NI, m))
            dP = Su @ (g[:, None] * qi) + l2 * P * (np.asarray(Su.sum(1)).ravel() > 0)[:, None]
            dQ = Si @ (g[:, None] * pu) + l2 * Q * (np.asarray(Si.sum(1)).ravel() > 0)[:, None]
            dbu = Su @ g; dbi = Si @ g
            gP += dP * dP; gQ += dQ * dQ; gbu += dbu * dbu; gbi += dbi * dbi
            P -= lr * dP / np.sqrt(gP); Q -= lr * dQ / np.sqrt(gQ)
            bu -= lr * dbu / np.sqrt(gbu); bi -= lr * dbi / np.sqrt(gbi)
    return dict(b=b, bu=bu, bi=bi, P=P, Q=Q)

def build(C, T_ev, T_rows, TE, TR, NI, NU, with_mf=True, sv_mode='std'):
    t0 = time.time()
    P_ev = TE.filter(pl.col('ts') < C)
    P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    g = float(P_rows['clicked'].mean())
    ndays = (C - J20) / DAY
    CL = click_log(C, P_ev, P_rows, T_ev)          # per-user stats: train data + each target row's OWN history
    CLg = click_log(C, P_ev, P_rows, T_ev.head(0))  # item-level / global stats: train data only (no target-event histories)

    # ---------------- item ----------------
    def ictr(c, n):
        return (pl.col(c) + K_ITEM * g) / (pl.col(n) + K_ITEM)
    it = P_rows.group_by('ii').agg(
        pl.len().alias('it_imp'), pl.col('clicked').sum().cast(pl.Int32).alias('it_clk'),
        (pl.col('ts') >= C - DAY).sum().alias('it_imp24'), pl.col('clicked').filter(pl.col('ts') >= C - DAY).sum().cast(pl.Int32).alias('it_clk24'),
        (pl.col('ts') >= C - 6 * 3600).sum().alias('it_imp6'), pl.col('clicked').filter(pl.col('ts') >= C - 6 * 3600).sum().cast(pl.Int32).alias('it_clk6'))
    it = it.with_columns(ictr('it_clk', 'it_imp').alias('it_ctr'), ictr('it_clk24', 'it_imp24').alias('it_ctr24'),
                         ictr('it_clk6', 'it_imp6').alias('it_ctr6'))
    it = it.with_columns((pl.col('it_ctr24') / pl.col('it_ctr')).alias('it_trend24'), (pl.col('it_ctr6') / pl.col('it_ctr')).alias('it_trend6'),
                         (pl.col('it_imp24') / (pl.col('it_imp') / ndays)).alias('it_imptrend24'),
                         (pl.col('it_imp') / ndays).alias('it_imp_rate'))
    ip = CLg.group_by('ii').agg(pl.len().alias('it_nclk_all'), pl.col('u').n_unique().alias('it_nusers'),
                               (pl.col('ts') < J20).sum().alias('it_nclk_warm'),
                               ((pl.col('ts') - C) / (2 * DAY)).exp().sum().cast(pl.Float32).alias('it_clk_dec2d'))

    # ---------------- user ----------------
    ue = P_rows.group_by('u').agg(pl.col('eid').n_unique().alias('us_nev'), pl.col('clicked').sum().cast(pl.Int32).alias('us_nclk_win'),
                                  pl.col('ts').max().alias('_u_last_ev'))
    ue = ue.with_columns(((pl.col('us_nclk_win') + K_USER * 0.629) / (pl.col('us_nev') + K_USER)).alias('us_ctr'))
    uc = CL.group_by('u').agg(pl.len().alias('us_nclk_all'), (pl.col('ts') < J20).sum().alias('us_nclk_warm'),
                              pl.col('ii').n_unique().alias('us_nuniq'), pl.col('ts').max().alias('_u_last_clk'),
                              (pl.col('ts') >= C - DAY).sum().alias('us_nclk24'))
    uc = uc.with_columns((1 - pl.col('us_nuniq') / pl.col('us_nclk_all')).alias('us_repeat_rate'))

    # ---------------- user x item ----------------
    uic = CL.group_by('u', 'ii').agg(pl.len().alias('hist_n_all'), (pl.col('ts') < J20).sum().alias('hist_n_warm'),
                                     (pl.col('ts') >= J20).sum().alias('hist_n_win'), pl.col('ts').max().alias('_ui_last_clk'))
    uii = P_rows.group_by('u', 'ii').agg(pl.len().alias('hist_imp'), pl.col('clicked').sum().cast(pl.Int32).alias('_ui_clk'),
                                         pl.col('ts').max().alias('_ui_last_imp'))
    uii = uii.with_columns((pl.col('hist_imp') - pl.col('_ui_clk')).alias('hist_noclk'))

    # ---------------- assemble ----------------
    R = (T_rows.select([c for c in ('rowid', 'eid', 'ii', 'clicked', 'ord') if c in T_rows.columns])
         .join(T_ev.select('eid', 'u', 'ts', 'tb', 'dt', 'h', 'h_ts'), on='eid', how='left').with_row_index('r'))
    R = (R.join(it, on='ii', how='left').join(ip, on='ii', how='left').join(ue, on='u', how='left').join(uc, on='u', how='left')
          .join(uic, on=['u', 'ii'], how='left').join(uii, on=['u', 'ii'], how='left'))
    R = R.with_columns(
        pl.col('h').list.len().alias('us_hlen'),
        (pl.col('h').list.len() == 0).cast(pl.Int8).alias('us_nohist'),
        pl.col('us_nev').is_null().cast(pl.Int8).alias('us_cold'),
        (pl.col('ts') - pl.col('_u_last_ev')).alias('us_since_ev'),
        (pl.col('ts') - pl.col('_u_last_clk')).alias('us_since_clk'),
        (pl.col('ts') - pl.col('_ui_last_clk')).alias('hist_since_clk'),
        (pl.col('ts') - pl.col('_ui_last_imp')).alias('hist_since_imp'),
        (pl.col('ts') - C).alias('tm_stale'), ((pl.col('ts') // 3600) % 24).cast(pl.Int8).alias('tm_hour'),
        pl.col('tb').alias('tm_tb'), pl.col('dt').alias('tm_dt'))
    # position (from end, 1 = most recent) of candidate in own history
    hx = (R.select('r', 'ii', 'h').with_columns(pl.int_ranges(pl.col('h').list.len(), 0, -1).alias('pos'))
            .explode(['h', 'pos']).drop_nulls('h').filter(pl.col('h') == pl.col('ii'))
            .group_by('r').agg(pl.col('pos').min().alias('hist_pos'), pl.len().alias('hist_n_own')))
    R = R.join(hx, on='r', how='left')
    R = R.with_columns(pl.col('hist_since_clk').fill_null(30 * DAY), pl.col('hist_since_imp').fill_null(30 * DAY),
                       pl.col('hist_pos').fill_null(99), pl.col('hist_n_own').fill_null(0), pl.col('us_since_ev').fill_null(30 * DAY), pl.col('us_since_clk').fill_null(30 * DAY))
    R = R.with_columns((pl.col('hist_n_all').fill_null(0) > 0).cast(pl.Int8).alias('hist_in'))

    # ---------------- co-click ----------------
    co, trn = cooc_tables(CLg, NI)
    last20 = CL.group_by('u', maintain_order=True).agg(pl.col('ii').tail(20).alias('l20'))
    T_u = R.select('r', 'u', 'ii').join(last20, on='u', how='left')
    X = (T_u.with_columns(pl.int_ranges(pl.col('l20').list.len(), 0, -1).alias('pos')).explode(['l20', 'pos'])
            .drop_nulls('l20').rename({'l20': 'j'}).with_columns(pl.col('j').cast(pl.Int32))
            .join(co, on=['j', 'ii'], how='left').join(trn, on=['j', 'ii'], how='left')
            .with_columns(pl.col('cos').fill_null(0), pl.col('trp').fill_null(0)))
    l5 = pl.col('pos') <= 5
    agg = X.group_by('r').agg(
        pl.col('cos').max().alias('co_cos_max20'), pl.col('cos').mean().alias('co_cos_mean20'),
        pl.col('cos').filter(l5).max().alias('co_cos_max5'), pl.col('cos').filter(l5).mean().alias('co_cos_mean5'),
        pl.col('cos').filter(pl.col('pos') == 1).first().alias('co_cos_last'),
        pl.col('trp').max().alias('co_tr_max20'), pl.col('trp').mean().alias('co_tr_mean20'),
        pl.col('trp').filter(l5).max().alias('co_tr_max5'), pl.col('trp').filter(l5).mean().alias('co_tr_mean5'),
        pl.col('trp').filter(pl.col('pos') == 1).first().alias('co_tr_last'),
        (pl.col('cos') / pl.col('pos').cast(pl.Float32).sqrt()).sum().alias('co_cos_rw'), pl.col('cos').sum().alias('co_cos_sum20'))
    R = R.join(agg, on='r', how='left').sort('r')

    # ---------------- SVD embeddings ----------------
    # sv_mode (leakage checks only): 'std' = click log as of C; 'notgt' = drop target events' own histories;
    # 'lag1d' = only clicks/impressions with ts < C - 1 day
    if sv_mode == 'std':
        CLsv, NEGsrc = CLg, P_rows
    elif sv_mode == 'notgt':
        CLsv = pl.concat([P_rows.filter(pl.col('clicked') == 1).select('u', pl.col('ii').cast(pl.Int32), 'ts'),
                          _hist_long(P_ev).filter(pl.col('ts') < C)]).unique()
        NEGsrc = P_rows
    elif sv_mode == 'lag1d':
        CLsv, NEGsrc = CL.filter(pl.col('ts') < C - DAY), P_rows.filter(pl.col('ts') < C - DAY)
    else:
        raise ValueError(sv_mode)
    assert (CLsv['ts'].max() or 0) < C and (NEGsrc['ts'].max() or 0) < C
    cnt = CLsv.group_by('u', 'ii').agg(pl.len().alias('n'))
    Xc = sparse.csr_matrix((np.ones(cnt.height, np.float32), (cnt['u'].to_numpy(), cnt['ii'].to_numpy())), shape=(NU, NI))
    df_ = np.asarray(Xc.sum(0)).ravel(); idf = np.log((NU + 1) / (df_ + 1)).astype(np.float32)
    Xc = Xc @ sparse.diags(idf); rn = np.sqrt(np.asarray(Xc.multiply(Xc).sum(1)).ravel()) + 1e-6; Xc = sparse.diags(1 / rn) @ Xc
    # user-side matrix incl. each target user's own history (fold-in onto item factors learned from train-only clicks)
    cnt = CL.filter(pl.col('ts') < C).group_by('u', 'ii').agg(pl.len().alias('n')) if sv_mode == 'std' else cnt
    Xu = sparse.csr_matrix((np.ones(cnt.height, np.float32), (cnt['u'].to_numpy(), cnt['ii'].to_numpy())), shape=(NU, NI)) @ sparse.diags(idf)
    Xu = sparse.diags(1 / (np.sqrt(np.asarray(Xu.multiply(Xu).sum(1)).ravel()) + 1e-6)) @ Xu
    uu = R['u'].to_numpy().astype(np.int64); iic = R['ii'].to_numpy().astype(np.int64)
    ex = X.select('r', 'j', 'pos').unique(['r', 'j', 'pos'])
    # per-event history (last 20 clicks) is shared by all candidates: take rows of one candidate per (r) anyway
    er_r = ex['r'].to_numpy(); er_j = ex['j'].to_numpy().astype(np.int64); er_w = (1.0 / np.sqrt(ex['pos'].to_numpy())).astype(np.float32)
    er_l5 = ex['pos'].to_numpy() <= 5
    nR = R.height
    svcols = {}
    for k in (16, 32, 64):
        U, S, Vt = randomized_svd(Xc, k, random_state=0, n_iter=5)
        V = (Vt.T * np.sqrt(S)).astype(np.float32); Uf = np.asarray(Xu @ (Vt.T / np.sqrt(S))).astype(np.float32)  # fold-in = U*sqrt(S) for train users
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9); Un = Uf / (np.linalg.norm(Uf, axis=1, keepdims=True) + 1e-9)
        cand = Vn[iic]
        svcols[f'sv_k{k}_ucos'] = (Un[uu] * cand).sum(1)
        svcols[f'sv_k{k}_udot'] = (Uf[uu] * V[iic]).sum(1)
        hv = np.zeros((nR, k), np.float32); np.add.at(hv, er_r, Vn[er_j] * er_w[:, None])
        hn = np.linalg.norm(hv, axis=1); svcols[f'sv_k{k}_hcos'] = (hv * cand).sum(1) / (hn + 1e-9)
        if k == 32:
            svcols['sv_k32_hnorm'] = hn
            sims = (Vn[er_j] * Vn[iic[er_r]]).sum(1)
            mx = np.full(nR, -1.0, np.float32); np.maximum.at(mx, er_r, sims)
            mx5 = np.full(nR, -1.0, np.float32); np.maximum.at(mx5, er_r[er_l5], sims[er_l5])
            svcols['sv_k32_imax20'] = mx; svcols['sv_k32_imax5'] = mx5
        if k == 64:
            # click centroid vs non-clicked-impression centroid (impressions frozen at C: P_rows only)
            Sc = sparse.csr_matrix((np.ones(cnt.height, np.float32), (cnt['u'].to_numpy(), cnt['ii'].to_numpy())), shape=(NU, NI))
            neg = NEGsrc.filter(pl.col('clicked') == 0)
            Sn = sparse.csr_matrix((np.ones(neg.height, np.float32), (neg['u'].to_numpy(), neg['ii'].to_numpy())), shape=(NU, NI))
            nC = np.asarray(Sc @ Vn); nC /= np.linalg.norm(nC, axis=1, keepdims=True) + 1e-9
            nN = np.asarray(Sn @ Vn); nN /= np.linalg.norm(nN, axis=1, keepdims=True) + 1e-9
            cc = (nC[uu] * cand).sum(1); cn = (nN[uu] * cand).sum(1)
            svcols['sv_k64_ccos'] = cc; svcols['sv_k64_ncos'] = cn
            for a, nm in ((0.25, '025'), (0.5, '05'), (1.0, '1')):
                M = nC - a * nN; M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
                svcols[f'sv_k64_con{nm}'] = (M[uu] * cand).sum(1)
    R = R.with_columns([pl.Series(c, v.astype(np.float32)) for c, v in svcols.items()])

    # ---------------- MF ----------------
    if with_mf:
        mf = fit_mf(P_rows, NU, NI)
        uu = R['u'].to_numpy().astype(np.int64); ii = R['ii'].to_numpy().astype(np.int64)
        dot = (mf['P'][uu] * mf['Q'][ii]).sum(1)
        R = R.with_columns(pl.Series('mf_score', mf['b'] + mf['bu'][uu] + mf['bi'][ii] + dot),
                           pl.Series('mf_dot', dot), pl.Series('mf_bi', mf['bi'][ii]))

    fill0 = [c for c in R.columns if c.startswith(('it_imp', 'it_clk', 'it_n', 'us_n', 'hist_n', 'hist_imp', 'hist_noclk', 'co_')) and R[c].dtype != pl.List]
    R = R.with_columns([pl.col(c).fill_null(0) for c in fill0])
    R = R.with_columns([pl.col(c).fill_null(g) for c in ('it_ctr', 'it_ctr24', 'it_ctr6')] +
                       [pl.col(c).fill_null(1.0) for c in ('it_trend24', 'it_trend6', 'it_imptrend24')] +
                       [pl.col('us_ctr').fill_null(g), pl.col('it_imp_rate').fill_null(0), pl.col('us_repeat_rate').fill_null(0)])

    # ---------------- event context ----------------
    main = ['it_ctr', 'it_ctr24', 'it_imp', 'it_nclk_all', 'it_clk_dec2d', 'hist_n_all', 'hist_noclk',
            'co_cos_max20', 'co_cos_rw', 'co_tr_max5', 'co_tr_last',
            'sv_k16_ucos', 'sv_k32_ucos', 'sv_k64_ucos', 'sv_k32_hcos', 'sv_k64_hcos', 'sv_k32_imax20', 'sv_k64_con05', 'sv_k64_con025', 'sv_k64_ccos'] + (['mf_score', 'mf_dot'] if with_mf else [])
    exprs = []
    for c in main:
        x = pl.col(c).cast(pl.Float64)
        exprs += [x.rank('average').over('eid').alias(f'ev_{c}_rk'),
                  ((x - x.mean().over('eid')) / (x.std().over('eid') + 1e-9)).alias(f'ev_{c}_z'),
                  x.max().over('eid').alias(f'ev_{c}_max'), x.mean().over('eid').alias(f'ev_{c}_mean')]
    R = R.with_columns(exprs)
    R = R.drop([c for c in R.columns if c.startswith('_')] + ['h', 'h_ts', 'tb', 'dt']).sort('r').drop('r')
    log(f'build C={C} rows={R.height} cols={R.width} {time.time()-t0:.0f}s')
    return R

GROUPS = ['it', 'hist', 'us', 'co', 'ev', 'tm', 'sv', 'mf']

def feature_cols(df, groups):
    cols = []
    for c in df.columns:
        if c in ID_COLS: continue
        gname = c.split('_')[0]
        if gname == 'ev':
            src = c[3:].split('_')[0]  # ev_<src group>_...
            if 'ev' in groups and src in groups: cols.append(c)
        elif gname in groups:
            cols.append(c)
    return cols
