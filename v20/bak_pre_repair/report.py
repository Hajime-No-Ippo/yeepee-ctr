import json,os,pandas as pd
H='/workspace/huawei/v20'; o=['# v20 REPORT (auto-generated, incremental)\n']
def J(p): 
    p=f'{H}/{p}'; return json.load(open(p)) if os.path.exists(p) else None
r=J('out/ref.json')
if r: o.append('## Step 0 reference (484 feat, variant A, dwbig 1000r)\n```\n'+json.dumps(r,indent=1)+'\n```\n')
x=J('out/redundancy.json')
if x: o.append(f"## Step 1 prefilter: {x['n_in']} in, near-constant {len(x['near_constant'])}, redundant dropped {x['n_dropped_redundant']} in {x['n_clusters']} clusters, kept {x['n_kept']}\n")
a=J('av_flagged.json')
if a: o.append(f"## AV diagnostic (flag only): overall AUC {a['overall_av_auc']:.4f}, flagged {a['n_flagged']}\n```\n"+json.dumps(a['top20_by_gain'][:20],indent=0)[:3000]+'\n```\n')
s=J('out/stable_feats.json')
if s: o.append(f"## Step 3 stability: stable {len(s['stable'])}, unstable {s['n_unstable']}\n")
if os.path.exists(f'{H}/out/curve.md'): o.append('## Step 4 AUC vs N\n'+open(f'{H}/out/curve.md').read())
b=J('out/best_params.json')
if b: o.append('## Step 5 best params\n```\n'+json.dumps(b,indent=1)[:3000]+'\n```\n')
if os.path.exists(f'{H}/out/tiers.csv'): o.append('## Step 6 tiers\n```\n'+pd.read_csv(f'{H}/out/tiers.csv').to_string()+'\n```\n')
d=J('out/decision.json')
if d: o.append('## Decision\n```\n'+json.dumps(d,indent=1)+'\n```\n')
i=J('out/submission_info.json')
if i: o.append('## Submission (not uploaded)\n```\n'+json.dumps(i,indent=1)+'\n```\n')
open(f'{H}/REPORT.md','w').write('\n'.join(o))
