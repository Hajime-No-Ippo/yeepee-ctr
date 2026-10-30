import json,collections
base={('B',25):.81742,('B',23):.79647,('A',25):.81751,('A',23):.79679}
R=collections.defaultdict(dict)
for l in open('/workspace/huawei/v19cut/out/xlog.jsonl'):
    m=json.loads(l); v,st=m['tag'].split('_'); R[(v,st)][(m['cut'],m['seed'])]=m
out=['# v19cut: feature drop-set screen (dwbig LightGBM, 1000 rounds, lr .015, 4 threads pinned to cores 4-7, shared box)\n',
'Baseline seed 0 (from v19 logs, 8 threads, free box): B C25 .81742 C23 .79647; A C25 .81751 C23 .79679.\n',
'| var | set | nf (dropped) | C25 s0 | C25 s1 | C23 s0 | C23 s1 | dC25 | dC23 | fit_s C25 (s0/s1) | fit_s C23 |','|---|---|---|---|---|---|---|---|---|---|---|']
def auc(m): return m['auc']['1000'] if m else None
for (v,st),d in sorted(R.items()):
    g=lambda c,s:auc(d.get((c,s)))
    vals={k:g(*k) for k in [(25,0),(25,1),(23,0),(23,1)]}
    nf=[m['nf'] for m in d.values()][0]
    # baseline reference: mean over available baseline seeds for this var
    def mean(c): 
        x=[vals[(c,s)] for s in (0,1) if vals[(c,s)] is not None]; return sum(x)/len(x) if x else None
    bm={}
    for c in (25,23):
        b=R.get((v,'base'),{}); x=[auc(b[(c,s)]) for s in (0,1) if (c,s) in b]
        if (c,0) not in b: x.append(base[(v,c)])
        bm[c]=sum(x)/len(x) if x else None
    f=lambda x:'-' if x is None else f'{x:.5f}'
    dl=lambda c:'-' if mean(c) is None or bm[c] is None else f'{mean(c)-bm[c]:+.5f}'
    ft=lambda c:'/'.join(str(d[(c,s)]['fit_s']) if (c,s) in d else '-' for s in (0,1))
    out.append(f"| {v} | {st} | {nf} ({484-nf}) | {f(vals[(25,0)])} | {f(vals[(25,1)])} | {f(vals[(23,0)])} | {f(vals[(23,1)])} | {dl(25)} | {dl(23)} | {ft(25)} | {ft(23)} |")
out.append('\nDeltas = mean over available seeds of the set minus mean over baseline seeds (seed0 from v19 log + seed1 rerun here). Keep rule: both deltas >= -.00003.')
open('/workspace/huawei/v19cut/README.md','w').write('\n'.join(out)+'\n')
