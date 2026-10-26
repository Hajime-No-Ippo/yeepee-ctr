import json,sys
B={('i',25):0.79734,('i',26):0.80834,('gap',25):0.79579,('gap',26):0.8049}
R={}
for l in open('/workspace/huawei/v2work/screen_results.jsonl'):
    r=json.loads(l)
    if r['tag'].startswith(('v7_','gap_v7_')): R[(r['tag'],r['cut'])]=max(r['curve'].values())
ok={}
for v in ('AB','A','B'):
    d=[]
    for kind,pfx in (('i','v7_'),('gap','gap_v7_')):
        for c in (25,26):
            x=R.get((pfx+v,c)); d.append(None if x is None else round(x-B[(kind,c)],5))
    ok[v]=d; print(v,'deltas std25,std26,gap25,gap26:',d)
best=[v for v in ('AB','A','B') if all(x is not None and x>=0.002 for x in ok[v])]
print('PASS',best[0] if best else 'none')
sys.exit(0 if best else 3)
