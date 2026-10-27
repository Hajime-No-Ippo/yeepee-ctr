import sys,json
for l in open(sys.argv[1]):
    try:r=json.loads(l)
    except: print(l[:150]);continue
    print(r['cut'],r['group'].ljust(8),'d=%.5f'%r['delta'],'dv11=%.5f'%r['delta_vs_v11'],'CI[%.5f,%.5f]'%tuple(r['ci']),{k[2:]:round(v,4) for k,v in r.items() if k.startswith('d_')})
