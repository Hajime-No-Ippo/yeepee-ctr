import json;d=json.load(open('out/ablation.json'))
for s,v in d.items():
  for n,r in v.items():
    f=lambda k,m: (round(r[k][m],5) if r.get(k) and r[k].get(m) is not None else None)
    print(s,n,'B',r['best_iter25'],'n25',r['n_train25'],'v25',f('val25','auc'),'v25w',f('val25','auc_evalw'),'v25ev',f('val25','auc_event'),
      'v26',f('val26','auc'),'v26ev',f('val26','auc_event'),'v22',f('val22','auc'),'v22w',f('val22','auc_evalw'),'v22ev',f('val22','auc_event'),
      'b22',[f('val22','auc_<=1d'),f('val22','auc_1-2d'),f('val22','auc_>2d')],'b25',[f('val25','auc_<=1d'),f('val25','auc_1-2d')],'s',r['secs'])
