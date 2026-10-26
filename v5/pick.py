import json,sys,groups
R={}
for l in open('out/results.jsonl'):
    try: r=json.loads(l)
    except: continue
    if r.get('kind')=='screen': R[(r['tag'],r['cut'])]=r
a=lambda t,c: (lambda cv: cv[max(cv,key=int)])(R[(t,c)]['curve'])
d=lambda t,c: a(t,c)-a('base_v5',c)
mode=sys.argv[1]
if mode=='passed':
    print(','.join(g for g in ('A1','A2','A3') if (f'scr_{g}',25) in R and d(f'scr_{g}',25)>=0.0005 and d(f'scr_{g}',26)>=0))
elif mode=='final':   # groups for confirmation
    P=[g for g in ('A1','A2','A3') if d(f'scr_{g}',25)>=0.0005 and d(f'scr_{g}',26)>=0]
    if len(P)>1 and ('scr_Astack',26) in R and not (d('scr_Astack',25)>=0.0005 and d('scr_Astack',26)>=0):
        P=[max(P,key=lambda g:d(f'scr_{g}',25))]
    print(','.join(P))
