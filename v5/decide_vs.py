import json,sys
tag,ref=sys.argv[1],sys.argv[2]; R={}
for l in open('out/results.jsonl'):
    try: r=json.loads(l)
    except: continue
    if r.get('tag') in (tag,ref) and 'auc400_mean' in r: R[(r['tag'],str(r['cut']))]=r
ok=True; out={}
for c in ('25','26','23'):
    if (tag,c) not in R or (ref,c) not in R: print('missing',c); sys.exit(2)
    d=R[(tag,c)]['auc400_mean']-R[(ref,c)]['auc400_mean']; out[c]=round(d,5); ok&=d>=0.001
    print(c,ref,R[(ref,c)]['auc400_mean'],tag,R[(tag,c)]['auc400_mean'],'d=%+.5f'%d,'ev',R[(ref,c)]['event_auc_mean'],'->',R[(tag,c)]['event_auc_mean'])
wd=R[(tag,'23')]['seg_mean']['weekday']-R[(ref,'23')]['seg_mean']['weekday']; ok&=wd>=0
print('val23 weekday d=%+.5f'%wd); print('DECISION','PASS' if ok else 'FAIL')
open('out/results.jsonl','a').write(json.dumps(dict(step='decision',kind='decision',tag=tag,ref=ref,delta=out,weekday_delta=round(wd,5),pass_=bool(ok)))+'\n')
sys.exit(0 if ok else 1)
