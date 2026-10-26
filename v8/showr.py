import json,sys
B={}
for l in open('screen_results.jsonl'):
  r=json.loads(l); t=r['tag']
  if t.startswith(('base','i_','gap_','c_')):
    b=max(r['curve'].values()); key=('gap' if t.startswith('gap_') else 'std',r['cut'])
    if t in('base','gap_base'): B[key]=b
    d=b-B.get(key,b); print(r['cut'],t.ljust(16),r['n_feats'],r['curve'],'best',b,'d%+.5f'%d,'ev',r['event_auc'],'wi',r['within_auc'])
