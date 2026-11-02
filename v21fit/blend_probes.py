#!/usr/bin/env python3
"""Rebuild the v21fit blend probes (Q1,Q2,R1,S1,S2) from the rank arrays in the private cache dataset.
Needs only numpy + scipy.  See BLEND_RECIPE.md.
usage: python blend_probes.py --inputs DIR/v21fit_blend_inputs --rowid-from SOME_PROBE_OR_W10.csv.zip --out OUTDIR [--only Q1,R1]
rowid order = order of the rows in any v21fit probe zip (= submission_blend_newbest_v21_w10_20261031.csv.zip = merged.parquet order)."""
import argparse, csv, hashlib, io, os, zipfile
import numpy as np
from scipy.stats import rankdata
EXPECT_ROWID_SHA='f4a805a3d6d3c60396dfa9ae915c2c20590fc043011a9621b6b5579f8e302742'
nr=lambda x:(rankdata(x)-1)/(len(x)-1)          # average-rank, scaled to [0,1]

def blend(a,p,key):   # p=(c,g,q,v8,v5): EN - c*E0 + g*(1-q)*A + g*q*D + v8*v8regen + v5*v5
    c,g,q,v8,v5=p; EN,E0,A,D,V8,V5=(a[k] for k in ('EN','E0','A_all','D_dcn','v8regen','v5'))
    # operation order mirrors the original mkQ.py / mkR.py / mkS.py so that float rounding (and hence near-tie ranks) is the same
    if key in ('Q1','R1'):
        Q1=EN-.136*E0+.098*(.5*A+.5*D)
        return Q1 if key=='Q1' else Q1-.03*V8-.015*V5
    if key=='Q2': return EN-.136*E0+.098*D
    c,g,q,v8,v5=map(np.float64,p)   # numpy scalars (as loaded from pS*.npy originally) => D float32 is promoted to float64 here
    return EN-c*E0+g*(1-q)*A+g*q*D+v8*V8+v5*V5
PROBES={  # name -> params (c, g, q, v8, v5); v8/v5 weights are ADDED (negative = subtracted)
 'Q1':('probe_v21fit_Q1_EBnew_m136E0_p098_A50_DCN50',(.136,.098,.5,0,0)),
 'Q2':('probe_v21fit_Q2_EBnew_m136E0_p098DCN',       (.136,.098,1.0,0,0)),
 'R1':('probe_v21fit_R1_Q1_m030v8regen_m015v5',      (.136,.098,.5,-.03,-.015)),
 'S1':('probe_v21fit_S1_R1step_c109_g123_q67_v8m052_v5m033',(0.10902810947726392,0.12251977564719992,0.665,-0.052490334215038625,-0.033138516122339604)),
 'S2':('probe_v21fit_S2_R1step_c097_g129_q75_v8m079_v5m015',(0.09693596102582225,0.1285964186161041,0.75,-0.078587083614776,-0.015)),
}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',default='v21fit/blend_inputs'); ap.add_argument('--rowid-from',required=True)
    ap.add_argument('--out',default='.'); ap.add_argument('--only',default=','.join(PROBES)); a=ap.parse_args()
    z=zipfile.ZipFile(a.rowid_from); rows=list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]),'utf-8')))
    assert rows[0][0]=='rowid'; rid=[r[0] for r in rows[1:]]; assert len(rid)==529390
    h=hashlib.sha256('\n'.join(rid).encode()).hexdigest(); assert h==EXPECT_ROWID_SHA,'row order differs from merged.parquet order: '+h
    L=lambda n: np.load(os.path.join(a.inputs,n+'.npy'))   # keep native dtype (D_dcn is float32, as in the original run)
    arr={k:L(k) for k in ('EN','E0','A_all','D_dcn','v8regen','v5')}
    for k,v in arr.items(): assert v.shape==(529390,),k
    os.makedirs(a.out,exist_ok=True)
    for key in a.only.split(','):
        nm,p=PROBES[key]; r=nr(blend(arr,p,key)); assert np.isfinite(r).all() and r.min()>=0 and r.max()<=1
        csvp=os.path.join(a.out,nm+'.csv')
        with open(csvp,'w',newline='') as f:
            w=csv.writer(f,lineterminator='\n'); w.writerow(['rowid','predicted_probability'])
            for i,x in zip(rid,r): w.writerow([i,format(x,'.12f')])
        zp=csvp+'.zip'
        with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as zf: zf.write(csvp,arcname=os.path.basename(csvp))
        os.remove(csvp); print(nm,'->',zp)
if __name__=='__main__': main()
