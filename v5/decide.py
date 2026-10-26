import json,sys
tag=sys.argv[1]; B=json.load(open('out/base.json'))
R={}
for l in open('out/results.jsonl'):
    try: r=json.loads(l)
    except: continue
    if r.get('tag')==tag and 'auc400_mean' in r: R[str(r['cut'])]=r
ok=True; out={}
for c in ('25','26','23'):
    if c not in R: print('missing',c); sys.exit(2)
    d=R[c]['auc400_mean']-B[c]['auc400_mean']; out[c]=round(d,5); ok&=d>=0.001
    print(c,'base',B[c]['auc400_mean'],tag,R[c]['auc400_mean'],'d=%+.5f'%d,'ev',B[c]['event_auc_mean'],'->',R[c]['event_auc_mean'])
wd=R['23']['seg_mean']['weekday']-B['23']['seg_mean']['weekday']; ok&=wd>=0
print('val23 weekday',B['23']['seg_mean']['weekday'],'->',R['23']['seg_mean']['weekday'],'d=%+.5f'%wd)
print('DECISION','PASS' if ok else 'FAIL')
open('out/results.jsonl','a').write(json.dumps(dict(step='decision',kind='decision',tag=tag,delta=out,weekday_delta=round(wd,5),pass_=bool(ok)))+'\n')
sys.exit(0 if ok else 1)
