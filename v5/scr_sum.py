import json
R={}
for l in open('out/results.jsonl'):
    try: r=json.loads(l)
    except: continue
    if r.get('kind')=='screen' and r.get('step','').startswith(('A','B')): R[(r['tag'],r['cut'])]=r
def auc(r): c=r['curve']; return c[max(c,key=int)]
b={c:auc(R[('base_v5',c)]) for c in (25,26) if ('base_v5',c) in R}
print('base',b)
for (t,c),r in sorted(R.items()):
    if t=='base_v5': continue
    print(t,c,auc(r),'d=%+.5f'%(auc(r)-b.get(c,float('nan'))),'ev',r.get('event_auc'),'n',r['n_feats'])
