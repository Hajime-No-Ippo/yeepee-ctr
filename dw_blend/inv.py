import pandas as pd, re
s=pd.read_csv('scan_summary.csv')
pub={'erbrown/submission_best':'0.824558 (E0)','v21fit/S1':'0.829101','ctl6_B':'~0.8187 (v21 B variants; slate_B_p33_w80 0.818737)','coAp':'0.818270 (A)','coBp':'0.818302 (B)','v15':'0.817848 (control)'}
def trans(k):
    return bool(re.search(r'(^|[_/])(co|nm)?B[ps]?(_|\b)|_B_|listB|slateB|ctl6_B|v21diag/v21listB|B_c',k)) 
rows=[]
for _,r in s.iterrows():
    k=r['key']; pv=next((v for kk,v in pub.items() if kk in k),'')
    rows.append(dict(model=k,eval_path=('present (n27=%d)'%r.n27) if r.n27>0 else '',val_paths='C23:%d C25:%d files'%(r.n23,r.n25),cutoffs=('23 ' if r.n23>0 else '')+('25 ' if r.n25>0 else '')+('27(eval) ' if r.n27>0 else ''),auc_c23=r.auc23,auc_c25=r.auc25,public_score=pv,transductive='yes (B variant, eval-history graphs)' if trans(k) else 'no/unknown'))
pd.DataFrame(rows).to_csv('inventory.csv',index=False)
# extras with no fold preds
ex=[('erbrown/EN (latest teammate rank)','erbrown/(latest)','none','27 only','','','0.828856','no (teammate)'),('erbrown/submission_best.csv.zip (E0)','erbrown/submission_best.csv.zip','none','27 only','','','0.824558','no (teammate)'),('v21fit S1 blend','v21fit/out/probe_v21fit_S1_...csv.zip','none','27 only','','','0.829101','partly (A/B inputs)')]
pd.concat([pd.read_csv('inventory.csv'),pd.DataFrame(ex,columns=pd.read_csv('inventory.csv').columns)]).to_csv('inventory.csv',index=False)
print(len(pd.read_csv('inventory.csv')))
