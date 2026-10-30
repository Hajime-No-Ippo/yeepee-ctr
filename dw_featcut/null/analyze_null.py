import json,re,sys,numpy as np,glob
OUT='/workspace/huawei/dw_featcut/null'
real=json.load(open(f'{OUT}/imp_real.json')); nulls=[json.load(open(f)) for f in sorted(glob.glob(f'{OUT}/imp_null*.json'))]; K=len(nulls)
cols=[c for c in real['gain'] if re.match(r'^(v2_|ev_)',c)]
tot=sum(real['gain'].values()); ntot=[sum(n['gain'].values()) for n in nulls]
rows=[]
for c in cols:
    r=real['gain'][c]; ng=np.array([n['gain'][c] for n in nulls]); ns=np.array([n['split'][c] for n in nulls])
    rows.append(dict(col=c,fam='v2_' if c.startswith('v2_') else 'ev_',real=r,real_share=r/tot,real_split=real['split'][c],null_mean=ng.mean(),null_p75=np.percentile(ng,75),null_p90=np.percentile(ng,90),null_max=ng.max(),
        ratio_mean=r/max(ng.mean(),1e-9),score=float(np.log(1e-10+r/(1+np.percentile(ng,75)))),pct_null_below=float((ng<r).mean()*100),null_split_mean=ns.mean()))
rows.sort(key=lambda d:d['real'])
json.dump(dict(K=K,total_real_gain=tot,total_null_gain=ntot,rows=rows),open(f'{OUT}/null_table.json','w'),indent=1)
print('K',K,'total real gain %.3g, null totals'%tot,[f'{x:.3g}' for x in ntot])
def cnt(name,f):
    s=[d for d in rows if f(d)]; print(f"{name}: {len(s)} (v2_ {sum(d['fam']=='v2_' for d in s)}, ev_ {sum(d['fam']=='ev_' for d in s)}) e.g.",[d['col'] for d in s[:6]],'real_share_sum=%.4f'%sum(d['real_share'] for d in s))
cnt('a real<null_p90',lambda d:d['real']<d['null_p90']); cnt('b real<null_mean',lambda d:d['real']<d['null_mean'])
for m in (2,3,5,10,20): cnt(f'ratio<{m}',lambda d,m=m:d['ratio_mean']<m)
cnt('pct_null_below<100',lambda d:d['pct_null_below']<100); cnt('score<0',lambda d:d['score']<0)
print('lowest ratios:'); [print(f"  {d['col']:28s} real {d['real']:.3g} null_mean {d['null_mean']:.3g} p90 {d['null_p90']:.3g} ratio {d['ratio_mean']:.1f} share {d['real_share']:.5f}") for d in sorted(rows,key=lambda d:d['ratio_mean'])[:25]]
