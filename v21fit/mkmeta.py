import polars as pl
d=pl.scan_csv('../data/evaluation.csv',schema_overrides={'timestamp':pl.Int64}).select(['rowid','event_id','user_id','item_id','timestamp','time_bucket','day_type','history_timestamps']).with_columns(
 pl.col('history_timestamps').str.extract_all(r'\d+').list.eval(pl.element().cast(pl.Int64)).alias('ht')).with_columns(
 pl.col('ht').list.len().alias('hlen'),(pl.col('timestamp')-pl.col('ht').list.max()).alias('stale_s'),
 (pl.col('timestamp')-pl.col('ht').list.min()).alias('span_s')).drop(['history_timestamps','ht']).collect()
d.write_parquet('meta.parquet'); print(d.shape); print(d.describe())
