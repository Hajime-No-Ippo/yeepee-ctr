"""Feature builder for TechArena CTR. All stats for a cutoff C use only events with ts < C
(plus the target events' own histories, which by construction contain only clicks < their ts)."""
import polars as pl, numpy as np, time
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

DAY = 86400
K_SMOOTH = 20

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def load(D):
    TE = pl.read_parquet(f'{D}/train_events.parquet')
    TR = pl.read_parquet(f'{D}/train_rows.parquet')
    EE = pl.read_parquet(f'{D}/evaluation_events.parquet')
    ER = pl.read_parquet(f'{D}/evaluation_rows.parquet')
    # global item index
    items = pl.concat([TR['item'], ER['item'], TE['h_items'].explode().drop_nulls(), EE['h_items'].explode().drop_nulls()]).unique().sort()
    imap = pl.DataFrame({'item': items, 'ii': pl.arange(0, len(items), eager=True).cast(pl.Int32)})
    def mapi(df, col='item'):
        return df.join(imap.rename({'item': col, 'ii': col + '_ii'}), on=col, how='left')
    TR = mapi(TR).drop('item').rename({'item_ii': 'ii'})
    ER = mapi(ER).drop('item').rename({'item_ii': 'ii'})
    def map_hist(E):
        ex = E.select('eid', 'h_items').explode('h_items').with_row_index('r')
        ex = ex.join(imap.rename({'item': 'h_items'}), on='h_items', how='left').sort('r')
        h = ex.group_by('eid', maintain_order=True).agg(pl.col('ii').drop_nulls().alias('h'))
        return E.drop('h_items').join(h, on='eid', how='left').with_columns(pl.col('h').fill_null([]))
    TE = map_hist(TE); EE = map_hist(EE)
    return TE, TR, EE, ER, len(items)

def first_events(TE, C, W):
    """first event per user in [C, C+W)"""
    return (TE.filter((pl.col('ts') >= C) & (pl.col('ts') < C + W)).sort('ts')
              .group_by('user', maintain_order=True).first())

def build(C, T_ev, T_rows, TE, TR, NI, svd_k=64, with_label=True):
    t0 = time.time()
    P_ev = TE.filter(pl.col('ts') < C)
    P_rows = TR.join(P_ev.select('eid', 'user', 'ts', 'tb'), on='eid', how='inner')
    g = P_rows['clicked'].mean() if len(P_rows) else 0.063
    ndays = max((C - P_ev['ts'].min()) / DAY, 1e-3) if len(P_ev) else 1e-3

    # ---------- click pool (user, item, ts) ----------
    def hist_clicks(E):
        return (E.select('user', 'h', 'h_ts').explode(['h', 'h_ts']).drop_nulls()
                 .rename({'h': 'ii', 'h_ts': 'ts'}))
    pool = pl.concat([hist_clicks(P_ev), hist_clicks(T_ev),
                      P_rows.filter(pl.col('clicked') == 1).select('user', 'ii', 'ts')]).unique()
    pool = pool.with_columns(pl.col('ii').cast(pl.Int32), pl.col('ts').cast(pl.Int64))
    # global/item-level pool: train data only (no target-event histories, so nothing aggregates across eval rows);
    # `pool` (with each target row's own history) is used only for that user's own features
    poolg = pl.concat([hist_clicks(P_ev), P_rows.filter(pl.col('clicked') == 1).select('user', 'ii', 'ts')]).unique() \
              .with_columns(pl.col('ii').cast(pl.Int32), pl.col('ts').cast(pl.Int64))
    log('pool', len(pool))

    # ---------- item stats ----------
    it = P_rows.group_by('ii').agg(pl.len().alias('i_imp'), pl.col('clicked').sum().alias('i_clk'),
                                   (pl.col('ts') >= C - DAY).sum().alias('i_imp1d'),
                                   pl.col('clicked').filter(pl.col('ts') >= C - DAY).sum().alias('i_clk1d'))
    it = it.with_columns(((pl.col('i_clk') + K_SMOOTH * g) / (pl.col('i_imp') + K_SMOOTH)).alias('i_ctr'),
                         ((pl.col('i_clk1d') + K_SMOOTH * g) / (pl.col('i_imp1d') + K_SMOOTH)).alias('i_ctr1d'),
                         (pl.col('i_imp') / ndays).alias('i_imp_rate'))
    ip = poolg.group_by('ii').agg(pl.len().alias('ip_n'), pl.col('user').n_unique().alias('ip_users'),
                                 (pl.col('ts') >= C - DAY).sum().alias('ip_1d'),
                                 (pl.col('ts') >= C - 3 * DAY).sum().alias('ip_3d'),
                                 ((pl.col('ts') - C) / DAY).exp().sum().alias('ip_dec1'),
                                 ((pl.col('ts') - C) / (3 * DAY)).exp().sum().alias('ip_dec3'),
                                 (pl.col('ts') < 1737331200).sum().alias('ip_warm'))
    itb = P_rows.group_by('ii', 'tb').agg(pl.len().alias('n'), pl.col('clicked').sum().alias('c'))
    itb = itb.with_columns(((pl.col('c') + K_SMOOTH * g) / (pl.col('n') + K_SMOOTH)).alias('i_ctr_tb')).select('ii', 'tb', 'i_ctr_tb')

    # ---------- user stats ----------
    ue = P_rows.group_by('user').agg(pl.col('eid').n_unique().alias('u_ev'), pl.col('clicked').sum().alias('u_clk'),
                                     pl.col('ts').max().alias('u_last_ev'))
    ue = ue.with_columns(((pl.col('u_clk') + 5 * g * 10) / (pl.col('u_ev') + 5)).alias('u_ctr'))
    up = pool.group_by('user').agg(pl.len().alias('up_n'), pl.col('ts').max().alias('up_last'),
                                   pl.col('ii').n_unique().alias('up_uniq'),
                                   (pl.col('ts') >= C - DAY).sum().alias('up_1d'))
    # user-item prior impressions
    ui = P_rows.group_by('user', 'ii').agg(pl.len().alias('ui_imp'), pl.col('clicked').sum().alias('ui_clk'),
                                           pl.col('ts').max().alias('ui_last_imp'))
    uip = pool.group_by('user', 'ii').agg(pl.len().alias('uip_n'), pl.col('ts').max().alias('uip_last'))

    # ---------- co-occurrence (user sets) & transitions ----------
    us = poolg.select('user', 'ii').unique()
    # v8: user-set co-occurrence counts via sparse B^T B (same integer counts as the former self-join, far less memory)
    _ur = us['user'].rank('dense').to_numpy().astype(np.int64) - 1
    _B = sparse.csr_matrix((np.ones(len(us), np.int32), (_ur, us['ii'].to_numpy().astype(np.int64))), shape=(int(_ur.max()) + 1 if len(us) else 0, NI))
    _C = (_B.T @ _B).tocsr(); _pop = _C.diagonal().astype(np.float64); _C.setdiag(0); _C.eliminate_zeros()
    del _B, _ur
    log('cooc pairs', _C.nnz)  # _C / _pop replace the (ii, ii_b, co, cos) frame + hash join
    seq = poolg.sort('user', 'ts')
    tr_pairs = []
    for lag in (1, 2, 3):
        tr_pairs.append(seq.with_columns(pl.col('ii').shift(lag).over('user').alias('prev'))
                           .drop_nulls('prev').select(pl.col('prev').cast(pl.Int32), 'ii', pl.lit(1.0 / lag).alias('w')))
    trn = pl.concat(tr_pairs).group_by('prev', 'ii').agg(pl.col('w').sum().alias('trn'))
    trn_out = trn.group_by('prev').agg(pl.col('trn').sum().alias('trn_tot'))
    trn = trn.join(trn_out, on='prev').with_columns((pl.col('trn') / (pl.col('trn_tot') + 5)).cast(pl.Float32).alias('trp')).select('prev', 'ii', 'trn', 'trp')

    # ---------- SVD embeddings of user x item click matrix ----------
    uidx = us.select('user').unique().with_row_index('ur')
    m = us.join(uidx, on='user')
    df_ = np.bincount(m['ii'].to_numpy(), minlength=NI)
    idf = np.log((len(uidx) + 1) / (df_ + 1)).astype(np.float32)
    X = sparse.csr_matrix((idf[m['ii'].to_numpy()], (m['ur'].to_numpy(), m['ii'].to_numpy())), shape=(len(uidx), NI))
    rn = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-6
    X = sparse.diags(1 / rn) @ X
    U, S, Vt = randomized_svd(X, svd_k, random_state=0, n_iter=4)
    V = Vt.T.astype(np.float32)  # unscaled singular vectors (best in emb_exp)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)
    V32 = V[:, :32] / (np.linalg.norm(V[:, :32], axis=1, keepdims=True) + 1e-6)
    # fold-in of every target user's own pool (train clicks + own history) onto the train-only factors
    uidx = pool.select('user').unique().with_row_index('ur')
    mu = pool.select('user', 'ii').unique().join(uidx, on='user')
    Xu = sparse.csr_matrix((idf[mu['ii'].to_numpy()], (mu['ur'].to_numpy(), mu['ii'].to_numpy())), shape=(len(uidx), NI))
    Xu = sparse.diags(1 / (np.sqrt(np.asarray(Xu.multiply(Xu).sum(1)).ravel()) + 1e-6)) @ Xu
    U = np.asarray(Xu @ (Vt.T / S)).astype(np.float32)
    S = np.ones_like(S)
    log('svd done')

    # ---------- assemble target rows ----------
    R = T_rows.join(T_ev, on='eid', how='left').with_row_index('r')
    R = R.with_columns((pl.col('ts') - C).alias('stale'), ((pl.col('ts') // 3600) % 24).cast(pl.Int8).alias('hour'))
    R = (R.join(it, on='ii', how='left').join(ip, on='ii', how='left').join(itb, on=['ii', 'tb'], how='left')
          .join(ue, on='user', how='left').join(up, on='user', how='left')
          .join(ui, on=['user', 'ii'], how='left').join(uip, on=['user', 'ii'], how='left').sort('r'))
    R = R.with_columns(
        pl.col('h').list.len().alias('h_len'),
        (pl.col('ts') - pl.col('h_ts').list.last()).alias('h_since_last'),
        (pl.col('ts') - pl.col('h_ts').list.first()).alias('h_since_first'),
        pl.col('h').list.n_unique().alias('h_uniq'),
        pl.col('h_ts').list.eval((pl.element() >= pl.element().max() - DAY).sum()).list.first().alias('h_1d_before_last'),
        (pl.col('ts') - pl.col('u_last_ev')).alias('u_since_ev'),
        (pl.col('ts') - pl.col('up_last')).alias('up_since'),
        (pl.col('ts') - pl.col('ui_last_imp')).alias('ui_since_imp'),
        (pl.col('ts') - pl.col('uip_last')).alias('uip_since'),
        (pl.col('ui_imp') - pl.col('ui_clk')).alias('ui_noclk'),
        (pl.col('ip_n') / (pl.col('i_imp') + 1)).alias('ip_per_imp'),
        (pl.col('i_ctr_tb') - pl.col('i_ctr')).alias('i_ctr_tb_diff'),
    )
    R = R.with_columns((pl.col('h_since_first') / (pl.col('h_len') - 1).clip(1)).alias('h_gap_mean'),
                       (pl.col('h_uniq') / pl.col('h_len').clip(1)).alias('h_uniq_ratio'))

    # pairwise candidate x history
    PR = (R.select('r', 'ii', 'h', 'h_ts', 'ts').with_columns(pl.int_ranges(pl.col('h').list.len(), 0, -1).alias('pos'))
            .explode(['h', 'h_ts', 'pos']).drop_nulls('h')
            .with_columns(pl.col('h').cast(pl.Int32)))
    _pi = PR['ii'].to_numpy().astype(np.int64); _ph = PR['h'].to_numpy().astype(np.int64)
    _cv = np.asarray(_C[_pi, _ph]).ravel().astype(np.float64)
    _cs = np.where(_cv > 0, _cv / np.sqrt(np.maximum(_pop[_pi] * _pop[_ph], 1e-12)), 0.0).astype(np.float32)
    PR = PR.with_columns(pl.Series('co', _cv.astype(np.int32)), pl.Series('cos', _cs)); del _pi, _ph, _cv, _cs
    PR = PR.with_columns((0.85 ** (pl.col('pos') - 1)).alias('rw'), (pl.col('h') == pl.col('ii')).alias('same'))
    agg = PR.group_by('r').agg(
        pl.col('cos').sum().alias('co_sum'), pl.col('cos').max().alias('co_max'), pl.col('cos').mean().alias('co_mean'),
        (pl.col('cos') * pl.col('rw')).sum().alias('co_rw'), pl.col('co').sum().alias('co_cnt'),
        pl.col('cos').filter(pl.col('pos') <= 3).sum().alias('co_last3'),
        pl.col('same').sum().alias('inh_cnt'), pl.col('pos').filter(pl.col('same')).min().alias('inh_pos'),
        (pl.col('ts') - pl.col('h_ts')).filter(pl.col('same')).min().alias('inh_since'),
        (pl.col('cos') > 0).sum().alias('co_nz'))
    R = R.join(agg, on='r', how='left')
    # transitions from last 3 history items
    TRp = (R.select('r', 'ii', pl.col('h').list.tail(3).alias('h3')).with_columns(pl.int_ranges(pl.col('h3').list.len(), 0, -1).alias('pos'))
             .explode(['h3', 'pos']).drop_nulls('h3').rename({'h3': 'prev'}).with_columns(pl.col('prev').cast(pl.Int32))
             .join(trn, on=['prev', 'ii'], how='left').with_columns(pl.col('trn').fill_null(0), pl.col('trp').fill_null(0)))
    tagg = TRp.group_by('r').agg(pl.col('trn').filter(pl.col('pos') == 1).sum().alias('tr_last'),
                                 pl.col('trp').filter(pl.col('pos') == 1).sum().alias('trp_last'),
                                 pl.col('trn').sum().alias('tr_last3'), pl.col('trp').sum().alias('trp_last3'))
    R = R.join(tagg, on='r', how='left').sort('r')
    # SVD cosine: recency weighted mean of history vectors
    ex = PR.select('r', 'h', 'rw')
    Wrh = sparse.csr_matrix((ex['rw'].to_numpy().astype(np.float32), (ex['r'].to_numpy(), ex['h'].to_numpy())), shape=(len(R), NI))  # v8: replaces np.add.at
    hv = np.asarray(Wrh @ V, dtype=np.float32)
    hvn = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-6)
    cand = V[R['ii'].to_numpy()]
    svd_cos = (hvn * cand).sum(1)
    # user factor from SVD (pool-based, covers clicks beyond last 20)
    uf = np.zeros((len(R), V.shape[1]), np.float32)
    um = R.select('user').join(uidx, on='user', how='left')['ur'].to_numpy()
    ok = ~np.isnan(um.astype(float)) if um.dtype.kind == 'f' else np.ones(len(um), bool)
    umi = np.nan_to_num(um.astype(float), nan=0).astype(np.int64)
    ufv = (U[umi] * np.sqrt(S)).astype(np.float32); ufv[~ok] = 0
    ufv /= (np.linalg.norm(ufv, axis=1, keepdims=True) + 1e-6)
    svd_ucos = (ufv * cand).sum(1)
    hv32 = np.asarray(Wrh @ V32, dtype=np.float32)
    svd32_cos = (hv32 / (np.linalg.norm(hv32, axis=1, keepdims=True) + 1e-6) * V32[R['ii'].to_numpy()]).sum(1)
    extra_svd = {}
    for kk in (16, 24, 48):
        Vk = V[:, :kk] / (np.linalg.norm(V[:, :kk], axis=1, keepdims=True) + 1e-6)
        hk = np.asarray(Wrh @ Vk, dtype=np.float32)
        extra_svd[f'svd{kk}_cos'] = (hk / (np.linalg.norm(hk, axis=1, keepdims=True) + 1e-6) * Vk[R['ii'].to_numpy()]).sum(1)
    # pairwise svd similarity: max over history, and with the most recent item
    pr_r = PR['r'].to_numpy(); pr_c = PR['ii'].to_numpy(); pr_h = PR['h'].to_numpy()
    sims = np.empty(len(PR), np.float32)
    for k in range(0, len(PR), 2_000_000):
        sl = slice(k, k + 2_000_000); sims[sl] = (V[pr_c[sl]] * V[pr_h[sl]]).sum(1)
    sp = pl.DataFrame({'r': pr_r, 'sim': sims, 'pos': PR['pos'].to_numpy(), 'same': PR['same'].to_numpy()}).filter(~pl.col('same'))
    sagg = sp.group_by('r').agg(pl.col('sim').max().alias('svd_maxsim'), pl.col('sim').filter(pl.col('pos') == 1).max().alias('svd_last'),
                                pl.col('sim').top_k(3).mean().alias('svd_top3'))
    R = R.with_columns(pl.Series('svd_cos', svd_cos), pl.Series('svd_ucos', svd_ucos), pl.Series('svd_hnorm', np.linalg.norm(hv, axis=1)),
                       pl.Series('svd32_cos', svd32_cos), *[pl.Series(k, v) for k, v in extra_svd.items()])
    R = R.join(sagg, on='r', how='left').sort('r')

    fill0 = ['i_imp', 'i_clk', 'i_imp1d', 'i_clk1d', 'ip_n', 'ip_users', 'ip_1d', 'ip_3d', 'ip_dec1', 'ip_dec3', 'ip_warm',
             'u_ev', 'u_clk', 'up_n', 'up_uniq', 'up_1d', 'ui_imp', 'ui_clk', 'ui_noclk', 'uip_n', 'co_sum', 'co_max', 'co_mean', 'co_rw',
             'co_cnt', 'co_last3', 'inh_cnt', 'co_nz', 'tr_last', 'trp_last', 'tr_last3', 'trp_last3']
    R = R.with_columns([pl.col(c).fill_null(0) for c in fill0])
    R = R.with_columns(pl.col('i_ctr').fill_null(g), pl.col('i_ctr1d').fill_null(g), pl.col('i_ctr_tb').fill_null(g), pl.col('u_ctr').fill_null(g * 10))
    # within-event relative features
    rel = ['i_ctr', 'ip_dec1', 'ip_n', 'co_sum', 'co_rw', 'co_max', 'svd_cos', 'svd_ucos', 'svd32_cos', 'svd16_cos', 'svd24_cos', 'svd48_cos', 'svd_maxsim', 'svd_top3', 'tr_last3', 'uip_n', 'i_ctr_tb', 'ui_noclk']
    R = R.with_columns([pl.col(c).rank('average').over('eid').alias(c + '_rk') for c in rel] +
                       [(pl.col(c) - pl.col(c).mean().over('eid')).alias(c + '_dm') for c in rel] +
                       [(pl.col(c) / (pl.col(c).max().over('eid') + 1e-9)).alias(c + '_rmax') for c in ['co_sum', 'svd_cos', 'ip_dec1', 'co_rw']])
    R = R.with_columns(pl.col('co_rw').max().over('eid').alias('ev_co_max'), pl.col('svd_cos').max().over('eid').alias('ev_svd_max'),
                       pl.col('svd_cos').mean().over('eid').alias('ev_svd_mean'),
                       (pl.col('inh_cnt') > 0).sum().over('eid').alias('ev_n_inh'), pl.col('i_ctr').mean().over('eid').alias('ev_ictr_mean'),
                       pl.col('ip_dec1').mean().over('eid').alias('ev_ipdec_mean'))
    log(f'build C={C} rows={len(R)} {time.time()-t0:.0f}s')
    starter = dict(g=g)
    return R

DROP = ['r', 'rowid', 'eid', 'user', 'ii', 'h', 'h_ts', 'ts', 'clicked', 'u_last_ev', 'up_last', 'ui_last_imp', 'uip_last', 'fold']
