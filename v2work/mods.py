"""Feature modifications for AUC-ideas-v2 screening (applied in cl.load_tab when variant contains '+mod' tokens).
All tables are train-only (TE/TR rows with ts < C, CLg train click log) or the row's own history. No eval-row aggregates.
  usfix : us_ctr recomputed with K_USER=12 and cold fill 0.629 (event-level prior) instead of 0.063
  share : cumulative counts -> share of the train-only global total at the same cutoff (+ item percentile rank); raw columns dropped
  notr  : drop noisy trend ratios (it_trend24/6, it_imptrend24, it_ctr24/6 and their ev_ aggregates)
  hs    : own-history shape (clicks 1h/6h/24h before event, last gap / median gap, gap CV, session clicks)"""
import sys, numpy as np, polars as pl
sys.path.insert(0, '/workspace/huawei/v3')
from features import load, click_log
from split import J20, DAY
_D = {}
def data():
    if not _D: _D['d'] = load('/workspace/huawei/v3/data')
    return _D['d']
def cutoff(n):
    if n == 'eval': return J20 + 7 * DAY
    if n.startswith('val'): return J20 + (int(n[3:]) - 20) * DAY
    return J20 + int(n[1:]) * DAY          # f/e/d/E k -> Ck
_T = {}
def totals(C):
    if C in _T: return _T[C]
    TE, TR, EE, ER, NI, NU = data()
    P_ev = TE.filter(pl.col('ts') < C); P_rows = TR.join(P_ev.select('eid', 'u', 'ts'), on='eid', how='inner')
    CLg = click_log(C, P_ev, P_rows, P_ev.head(0))
    ii = P_rows['ii'].to_numpy(); cl = P_rows['clicked'].to_numpy()
    imp = np.bincount(ii, minlength=NI).astype(float); clk = np.bincount(ii, weights=cl, minlength=NI)
    nca = np.bincount(CLg['ii'].to_numpy(), minlength=NI).astype(float)
    us = CLg.select('ii', 'u').unique(); nus = np.bincount(us['ii'].to_numpy(), minlength=NI).astype(float)
    warm = np.bincount(CLg.filter(pl.col('ts') < J20)['ii'].to_numpy(), minlength=NI).astype(float)
    def pct(a): r = a.argsort(kind='stable').argsort(); s = np.sort(a); return ((np.searchsorted(s, a, 'left') + np.searchsorted(s, a, 'right')) / 2 / len(a)).astype(np.float32)
    ue = P_rows.group_by('u').agg(pl.col('eid').n_unique().alias('nev'), pl.col('clicked').sum().alias('ncl'))
    uc = CLg.group_by('u').agg(pl.len().alias('nall'), pl.col('ii').n_unique().alias('nuq'))
    t = dict(imp=imp.sum(), clk=clk.sum(), nca=nca.sum(), nus=nus.sum(), warm=max(warm.sum(), 1), pimp=pct(imp), pclk=pct(clk), pnca=pct(nca),
             nev=ue['nev'].mean(), ncl=ue['ncl'].mean(), nall=uc['nall'].mean(), nuq=uc['nuq'].mean())
    _T[C] = t; return t
ITEM = {'imp': ['it_imp', 'v2_i_imp', 'ev_it_imp_max', 'ev_it_imp_mean'], 'clk': ['it_clk', 'v2_i_clk'],
        'nca': ['it_nclk_all', 'v2_ip_n', 'ev_it_nclk_all_max', 'ev_it_nclk_all_mean', 'v2_ip_n_dm'], 'nus': ['it_nusers', 'v2_ip_users'],
        'warm': ['it_nclk_warm', 'v2_ip_warm']}
USER = {'nev': ['us_nev', 'v2_u_ev'], 'ncl': ['us_nclk_win', 'v2_u_clk'], 'nall': ['us_nclk_all', 'v2_up_n'], 'nuq': ['us_nuniq', 'v2_up_uniq']}
SINCE = ['v2_ui_since_imp', 'us_since_ev', 'v2_u_since_ev', 'hist_since_imp']
def share(a, n):
    t = totals(cutoff(n)); ex = []
    for k, cs in ITEM.items():
        ex += [(pl.col(c) / t[k] * 1e5).cast(pl.Float32).alias('sh_' + c) for c in cs if c in a.columns]
    for k, cs in USER.items():
        ex += [(pl.col(c) / t[k]).cast(pl.Float32).alias('sh_' + c) for c in cs if c in a.columns]
    ex += [(pl.col(c) / (pl.col('ts') - J20)).cast(pl.Float32).alias('sh_' + c) for c in SINCE if c in a.columns]
    if 'v2_ip_per_imp' in a.columns: ex.append(((pl.col('v2_ip_per_imp') * t['imp'] / t['nca'])).cast(pl.Float32).alias('sh_v2_ip_per_imp'))
    ii = a['ii'].to_numpy()
    a = a.with_columns(ex + [pl.Series('sh_it_imp_gpct', t['pimp'][ii]), pl.Series('sh_it_clk_gpct', t['pclk'][ii]), pl.Series('sh_it_nca_gpct', t['pnca'][ii])])
    drop = [c for cs in list(ITEM.values()) + list(USER.values()) for c in cs] + SINCE + ['v2_ip_per_imp']
    return a.drop([c for c in drop if c in a.columns])
def usfix(a, n, K=12):
    return a.with_columns(((pl.col('us_nclk_win').fill_null(0) + K * 0.629) / (pl.col('us_nev').fill_null(0) + K)).cast(pl.Float32).alias('us_ctr'))
TREND = ['it_trend24', 'it_trend6', 'it_imptrend24', 'it_ctr24', 'it_ctr6', 'ev_it_ctr24_rk', 'ev_it_ctr24_z', 'ev_it_ctr24_max', 'ev_it_ctr24_mean']
def notr(a, n): return a.drop([c for c in TREND if c in a.columns])
def hs_table(E):
    H = 3600
    X = E.select('eid', 'ts', 'h_ts').explode('h_ts').drop_nulls('h_ts').with_columns(age=(pl.col('ts') - pl.col('h_ts')))
    agg = X.group_by('eid').agg((pl.col('age') <= H).sum().alias('hs_n1h'), (pl.col('age') <= 6 * H).sum().alias('hs_n6h'),
                                (pl.col('age') <= 24 * H).sum().alias('hs_n24h'))
    G = (E.select('eid', 'ts', pl.col('h_ts').list.sort().alias('t'))
          .with_columns(pl.col('t').list.diff().list.drop_nulls().alias('g'), (pl.col('ts') - pl.col('t').list.last()).alias('lg')))
    G = G.with_columns(pl.col('g').list.median().alias('gmed'), pl.col('g').list.mean().alias('gmean'), pl.col('g').list.std().alias('gstd'))
    # session = clicks chained backwards from the last click with gaps < 30 min (and last click within 30 min of the event)
    S = (E.select('eid', 'ts', pl.col('h_ts').list.sort(descending=True).alias('t')).explode('t').drop_nulls('t')
          .with_columns(pl.col('t').shift(1).over('eid').fill_null(pl.col('ts')).alias('nxt'))
          .with_columns(((pl.col('nxt') - pl.col('t')) >= 1800).cum_sum().over('eid').alias('brk'))
          .group_by('eid').agg((pl.col('brk') == 0).sum().alias('hs_sess')))
    out = (G.select('eid', (pl.col('lg') / (pl.col('gmed') + 60)).alias('hs_lg_med'), (pl.col('gstd') / (pl.col('gmean') + 60)).alias('hs_gcv'),
                    pl.col('gmed').alias('hs_gmed'), (pl.col('lg') / (pl.col('gmean') + 60)).alias('hs_lg_mean'))
             .join(agg, on='eid', how='left').join(S, on='eid', how='left'))
    return out.with_columns([pl.col(c).cast(pl.Float32) for c in out.columns if c != 'eid'])
_H = {}
def hs(a, n):
    k = 'eval' if n == 'eval' else 'train'
    if k not in _H:
        TE, TR, EE, ER, NI, NU = data(); _H[k] = hs_table(EE if k == 'eval' else TE)
    return a.join(_H[k], on='eid', how='left')
MODS = dict(usfix=usfix, share=share, notr=notr, hs=hs)
def apply(a, n, variant):
    for m in variant.split('+')[1:]: a = MODS[m](a, n)
    return a
