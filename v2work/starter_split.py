import csv, polars as pl, sys
csv.field_size_limit(10**9)
sys.path.insert(0,'/workspace/huawei/v2work')
D='/workspace/huawei/v2work'; C=1737331200+4*86400
te=pl.read_parquet(f'{D}/train_events.parquet',columns=['eid','user','ts'])
v=set(te.filter(pl.col('ts')>=C).sort('ts').group_by('user',maintain_order=True).first()['eid'].to_list())
with open(f'{D}/train.csv',newline='') as f, open(f'{D}/st_fit.csv','w',newline='') as a, open(f'{D}/st_val.csv','w',newline='') as b, open(f'{D}/st_val_labels.csv','w',newline='') as c:
    r=csv.reader(f); h=next(r); wa=csv.writer(a); wb=csv.writer(b); wc=csv.writer(c)
    wa.writerow(h); wb.writerow(h[:-1]); wc.writerow(['rowid','clicked'])
    for row in r:
        if int(row[4])<C: wa.writerow(row)
        elif int(row[1][4:19],16) in v: wb.writerow(row[:-1]); wc.writerow([row[0],row[-1]])
