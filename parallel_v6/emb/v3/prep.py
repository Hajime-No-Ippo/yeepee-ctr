"""Parse train/evaluation CSVs into compact parquet tables (dense int ids).

Outputs in v3/data/:
  rows_train.parquet  : rowid, eid, ii (item idx), clicked
  rows_eval.parquet   : rowid, eid, ii, ord (original row order)
  ev_train.parquet / ev_eval.parquet : eid, u (user idx), ts, tb, dt, h (list[int item idx]), h_ts (list[int])
  items.parquet / users.parquet : id string -> idx
Usage: python prep.py [DATA_DIR]
"""
import polars as pl, sys, time, os
DATA = sys.argv[1] if len(sys.argv) > 1 else '@@RAW@@'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.1f}s]', *a, flush=True)

def scan(name):
    return pl.scan_csv(f'{DATA}/{name}.csv', schema_overrides={'history_item_ids': pl.Utf8, 'history_timestamps': pl.Utf8,
                       'rowid': pl.Utf8, 'event_id': pl.Utf8, 'user_id': pl.Utf8, 'item_id': pl.Utf8})

raw = {}
for name, lab in (('train', True), ('evaluation', False)):
    lf = scan(name)
    cols = ['rowid', 'event_id', 'item_id'] + (['clicked'] if lab else [])
    rows = lf.select(cols).collect(engine='streaming')
    ev = (lf.select('event_id', 'user_id', 'timestamp', 'time_bucket', 'day_type', 'history_item_ids', 'history_timestamps')
            .unique(subset='event_id', keep='any')
            .with_columns(pl.col('history_item_ids').str.json_decode(pl.List(pl.Utf8)).alias('h_str'),
                          pl.col('history_timestamps').str.json_decode(pl.List(pl.Int64)).alias('h_ts'))
            .drop('history_item_ids', 'history_timestamps').collect(engine='streaming'))
    raw[name] = (rows, ev)
    log(name, rows.shape, ev.shape)

tr_rows, tr_ev = raw['train']; ev_rows, ev_ev = raw['evaluation']
items = pl.concat([tr_rows['item_id'], ev_rows['item_id'], tr_ev['h_str'].explode().drop_nulls(),
                   ev_ev['h_str'].explode().drop_nulls()]).unique().sort()
items = pl.DataFrame({'item_id': items, 'ii': pl.arange(0, len(items), eager=True).cast(pl.Int32)})
users = pl.concat([tr_ev['user_id'], ev_ev['user_id']]).unique().sort()
users = pl.DataFrame({'user_id': users, 'u': pl.arange(0, len(users), eager=True).cast(pl.Int32)})
events = pl.concat([tr_ev['event_id'], ev_ev['event_id']]).unique().sort()
events = pl.DataFrame({'event_id': events, 'eid': pl.arange(0, len(events), eager=True).cast(pl.Int32)})
log('items', len(items), 'users', len(users), 'events', len(events))
items.write_parquet(f'{OUT}/items.parquet'); users.write_parquet(f'{OUT}/users.parquet')

def map_rows(rows, lab):
    r = rows.with_row_index('ord').join(events, on='event_id', how='left').join(items, on='item_id', how='left')
    sel = ['ord', 'rowid', 'eid', 'ii'] + (['clicked'] if lab else [])
    r = r.sort('ord').select(sel)
    if lab: r = r.with_columns(pl.col('clicked').cast(pl.Int8))
    return r

def map_ev(ev):
    e = ev.join(events, on='event_id', how='left').join(users, on='user_id', how='left')
    ex = e.select('eid', 'h_str').with_columns(pl.int_ranges(0, pl.col('h_str').list.len()).alias('k')).explode(['h_str', 'k']).drop_nulls('h_str')
    ex = ex.join(items.rename({'item_id': 'h_str'}), on='h_str', how='left').sort('eid', 'k')
    h = ex.group_by('eid', maintain_order=True).agg(pl.col('ii').alias('h'))
    e = e.join(h, on='eid', how='left').with_columns(pl.col('h').fill_null(pl.lit([], dtype=pl.List(pl.Int32))))
    return e.select('eid', 'u', pl.col('timestamp').cast(pl.Int64).alias('ts'), pl.col('time_bucket').cast(pl.Int8).alias('tb'),
                    pl.col('day_type').cast(pl.Int8).alias('dt'), 'h', 'h_ts').sort('ts', 'eid')

map_rows(tr_rows, True).drop('ord').write_parquet(f'{OUT}/rows_train.parquet')
map_rows(ev_rows, False).write_parquet(f'{OUT}/rows_eval.parquet')
map_ev(tr_ev).write_parquet(f'{OUT}/ev_train.parquet')
map_ev(ev_ev).write_parquet(f'{OUT}/ev_eval.parquet')
log('done')
