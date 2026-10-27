"""v9 time features. python tm.py n1 n2 ...  (f1..f6, val23, val25, val26, eval)
Leak rules: item/user/user-item stats use TRAIN rows with ts < C (fold cutoff; C=Jan27 for eval), never eval rows.
Own-history features use the row's own h_ts strictly < row ts. Writes feat_tmA (decay/window), feat_tmB (time-of-day crosses), feat_tmB2 (raw time ids)."""
import sys, time, gc, numpy as np, polars as pl
D='/workspace/huawei/v9'; V8='/workspace/huawei/v8'; J20=1737331200; DAY=86400; H=3600; PRI=0.0629
CUT={**{f'f{k}':J20+k*DAY for k in range(1,7)},'val23':J20+3*DAY,'val25':J20+5*DAY,'val26':J20+6*DAY,'eval':J20+7*DAY}
HL={'1h':H,'6h':6*H,'1d':DAY,'3d':3*DAY}; WIN={'1h':H,'72h':72*H}; UWIN={'1h':H,'6h':6*H,'24h':24*H,'72h':72*H}
t0=time.time()
TE=pl.read_parquet(f'{V8}/train_events.parquet'); TR=pl.read_parquet(f'{V8}/train_rows.parquet')
RT=TR.select('eid','item','clicked').join(TE.select('eid','user','ts'),on='eid')
RT=RT.with_columns(hr=((pl.col('ts')%DAY)//H).cast(pl.Int8),dt=(((pl.col('ts')-J20)//DAY)>=5).cast(pl.Int8),tb=((pl.col('ts')%DAY)//(6*H)).cast(pl.Int8))
EV=pl.concat([TE,pl.read_parquet(f'{V8}/evaluation_events.parquet')]); ROWS=pl.concat([TR.select('rowid','eid','item'),pl.read_parquet(f'{V8}/evaluation_rows.parquet')])
print('load',round(time.time()-t0,1),flush=True)
def sm(c,i,m,p=PRI): return (c+p*m)/(i+m)
for n in sys.argv[1:]:
    t1=time.time(); C=CUT[n]
    T=pl.read_parquet(f'{D}/feat_v3/{n}.parquet',columns=['rowid']).join(ROWS,on='rowid',how='left',maintain_order='left')
    T=T.join(EV.select('eid','user','ts','h_items','h_ts'),on='eid',how='left',maintain_order='left')
    assert T['item'].null_count()==0 and T['ts'].null_count()==0 and (T['ts']>=C).all(), n
    T=T.with_columns(hr=((pl.col('ts')%DAY)//H).cast(pl.Int8),dt=(((pl.col('ts')-J20)//DAY)%7>=5).cast(pl.Int8),tb=((pl.col('ts')%DAY)//(6*H)).cast(pl.Int8),stale=(pl.col('ts')-C))
    P=RT.filter(pl.col('ts')<C).with_columns(age=(C-pl.col('ts')).cast(pl.Float64))
    # ---- A: item decay/window (as of C)
    ag=[pl.len().alias('imp'),pl.col('clicked').sum().alias('clk')]
    for k,w in WIN.items(): ag+=[(pl.col('age')<=w).sum().alias(f'imp{k}'),((pl.col('age')<=w)*pl.col('clicked')).sum().alias(f'clk{k}')]
    for k,h in HL.items():
        d=(2.0**(-pl.col('age')/h)); ag+=[d.sum().alias(f'dimp{k}'),(d*pl.col('clicked')).sum().alias(f'dclk{k}')]
    I=P.group_by('item').agg(ag)
    ex=[]
    for k in WIN: ex+=[pl.col(f'imp{k}').alias(f'tmA_i_imp{k}'),sm(pl.col(f'clk{k}'),pl.col(f'imp{k}'),20).alias(f'tmA_i_ctr{k}')]
    for k in HL: ex+=[pl.col(f'dimp{k}').alias(f'tmA_i_dimp{k}'),pl.col(f'dclk{k}').alias(f'tmA_i_dclk{k}'),sm(pl.col(f'dclk{k}'),pl.col(f'dimp{k}'),10).alias(f'tmA_i_dctr{k}')]
    I=I.select(['item',sm(pl.col('clk'),pl.col('imp'),20).alias('ictr'),pl.col('imp').alias('iimp')]+ex).with_columns(
        (pl.col('tmA_i_dctr6h')/pl.col('tmA_i_dctr3d')).log().alias('tmA_i_dtrend6h3d'),(pl.col('tmA_i_dctr1h')/pl.col('tmA_i_dctr1d')).log().alias('tmA_i_dtrend1h1d'),
        (pl.col('tmA_i_dimp1d')/(pl.col('tmA_i_dimp3d')+1)).alias('tmA_i_dimprat1d3d'))
    # ---- A: user decay/window (train events of the user before C)
    UE=P.group_by('eid').agg(pl.col('user').first(),pl.col('age').first(),pl.col('clicked').sum().alias('c'))
    ag=[]
    for k,w in UWIN.items(): ag+=[(pl.col('age')<=w).sum().alias(f'tmA_u_nev{k}'),((pl.col('age')<=w)*pl.col('c')).sum().alias(f'tmA_u_nclk{k}')]
    for k,h in HL.items():
        d=(2.0**(-pl.col('age')/h)); ag+=[d.sum().alias(f'tmA_u_dev{k}'),(d*pl.col('c')).sum().alias(f'tmA_u_dclk{k}')]
    U=UE.group_by('user').agg(ag)
    for k in ('6h','1d','3d'): U=U.with_columns(((pl.col(f'tmA_u_dclk{k}')+PRI*10*3)/(pl.col(f'tmA_u_dev{k}')*10+3)).alias(f'tmA_u_dctr{k}'))
    # ---- A: user x item decayed impressions/clicks (train rows before C)
    UI=P.join(T.select('user','item').unique(),on=['user','item'],how='semi').with_columns(d=2.0**(-pl.col('age')/DAY)).group_by('user','item').agg(
        pl.col('d').sum().alias('tmA_ui_dimp1d'),(pl.col('d')*pl.col('clicked')).sum().alias('tmA_ui_dclk1d'),(pl.col('age').min()).alias('tmA_ui_age_lastimp'))
    # ---- A: own history relative to row ts (h_ts < ts)
    Hx=T.select('rowid','item','ts','h_items','h_ts').explode('h_items','h_ts').filter(pl.col('h_ts').is_not_null()&(pl.col('h_ts')<pl.col('ts'))).with_columns(ga=(pl.col('ts')-pl.col('h_ts')).cast(pl.Float64))
    ag=[]
    for k,w in UWIN.items(): ag.append((pl.col('ga')<=w).sum().cast(pl.Float32).alias(f'tmA_h_n{k}'))
    for k,h in HL.items(): ag.append((2.0**(-pl.col('ga')/h)).sum().alias(f'tmA_h_d{k}'))
    same=pl.col('h_items')==pl.col('item')
    ag+=[pl.when(same).then(pl.col('ga')).min().alias('tmA_h_cand_since'),(same*(2.0**(-pl.col('ga')/DAY))).sum().alias('tmA_h_cand_d1d'),(same*(2.0**(-pl.col('ga')/(3*DAY)))).sum().alias('tmA_h_cand_d3d')]
    # B: user's history hour affinity
    hh=((pl.col('h_ts')%DAY)//H); rh=((pl.col('ts')%DAY)//H); dh=((hh-rh).abs()); dh=pl.min_horizontal(dh,24-dh)
    ag+=[(dh<=1).mean().alias('tmB_h_hraff1'),(dh<=3).mean().alias('tmB_h_hraff3'),((pl.col('h_ts')%DAY)//(6*H)==(pl.col('ts')%DAY)//(6*H)).mean().alias('tmB_h_tbaff'),
         ((((pl.col('h_ts')-J20)//DAY)%7>=5)==(((pl.col('ts')-J20)//DAY)%7>=5)).mean().alias('tmB_h_dtaff')]
    HF=Hx.group_by('rowid').agg(ag)
    # ---- B: item x hour / dt / dt*tb CTR (train rows before C), smoothed toward item ctr
    IH=P.group_by('item','hr').agg(pl.len().alias('n'),pl.col('clicked').sum().alias('c'))
    ID=P.group_by('item','dt').agg(pl.len().alias('n'),pl.col('clicked').sum().alias('c'))
    IDT=P.group_by('item','dt','tb').agg(pl.len().alias('n'),pl.col('clicked').sum().alias('c'))
    UT=UE.join(P.group_by('eid').agg(pl.col('tb').first()),on='eid').group_by('user','tb').agg(pl.len().alias('n'),pl.col('c').sum().alias('cc'))
    UN=UT.group_by('user').agg(pl.col('n').sum().alias('un'),pl.col('cc').sum().alias('uc'))
    GH=P.group_by('hr').agg(pl.col('clicked').mean().alias('g'))
    # ---- assemble
    R=T.select('rowid','eid','item','user','ts','hr','dt','tb','stale').join(I,on='item',how='left').join(U,on='user',how='left').join(UI,on=['user','item'],how='left').join(HF,on='rowid',how='left')
    R=R.join(IH.rename({'n':'ih_n','c':'ih_c'}),on=['item','hr'],how='left').join(ID.rename({'n':'id_n','c':'id_c'}),on=['item','dt'],how='left').join(IDT.rename({'n':'idt_n','c':'idt_c'}),on=['item','dt','tb'],how='left')
    R=R.join(UT,on=['user','tb'],how='left').join(UN,on='user',how='left').join(GH,on='hr',how='left')
    R=R.with_columns(pl.col('ictr').fill_null(PRI),*[pl.col(c).fill_null(0) for c in ['ih_n','ih_c','id_n','id_c','idt_n','idt_c','n','cc','un','uc']])
    R=R.with_columns(tmA_ui_age_lastimp=pl.col('tmA_ui_age_lastimp')+pl.col('stale').cast(pl.Float64),
        tmB_i_ctr_hr=sm(pl.col('ih_c'),pl.col('ih_n'),20,pl.col('ictr')),tmB_i_ctr_dt=sm(pl.col('id_c'),pl.col('id_n'),20,pl.col('ictr')),tmB_i_ctr_dttb=sm(pl.col('idt_c'),pl.col('idt_n'),20,pl.col('ictr')),
        tmB_u_tbshare=(pl.col('n')+1)/(pl.col('un')+4),tmB_u_ctr_tb=sm(pl.col('cc'),pl.col('n')*10,5,(pl.col('uc')+PRI*30)/(pl.col('un')*10+3)))
    R=R.with_columns(tmB_i_lift_hr=(pl.col('tmB_i_ctr_hr')/pl.col('ictr')).log(),tmB_i_lift_dt=(pl.col('tmB_i_ctr_dt')/pl.col('ictr')).log(),tmB_i_lift_dttb=(pl.col('tmB_i_ctr_dttb')/pl.col('ictr')).log(),
        tmB_i_hrshare=(pl.col('ih_n')+1)/(pl.col('iimp').fill_null(0)+24),tmB_g_hrctr=pl.col('g'))
    # within-event normalisations
    for c in ('tmA_i_dctr1d','tmA_i_dctr6h','tmA_i_dtrend6h3d','tmB_i_lift_hr','tmB_i_lift_dttb'):
        R=R.with_columns(pl.col(c).rank().over('eid').alias(c+'_rk'),(pl.col(c)-pl.col(c).mean().over('eid')).alias(c+'_dm'))
    A=[c for c in R.columns if c.startswith('tmA_')]; B=[c for c in R.columns if c.startswith('tmB_')]
    R.select(['rowid']+A).with_columns(pl.col(A).cast(pl.Float32)).write_parquet(f'{D}/feat_tmA/{n}.parquet')
    R.select(['rowid']+B).with_columns(pl.col(B).cast(pl.Float32)).write_parquet(f'{D}/feat_tmB/{n}.parquet')
    R.select('rowid',pl.col('hr').cast(pl.Float32).alias('tmR_hr'),pl.col('tb').cast(pl.Float32).alias('tmR_tb'),pl.col('dt').cast(pl.Float32).alias('tmR_dt'),(pl.col('dt')*24+pl.col('hr')).cast(pl.Float32).alias('tmR_dthr'),
             (pl.col('stale')/H).cast(pl.Float32).alias('tmR_stale_h')).write_parquet(f'{D}/feat_tmB2/{n}.parquet')
    nf=(Hx['h_ts']>=C).mean() if len(Hx) else 0
    print(n,R.shape,len(A),len(B),'hist>=C share',round(float(nf),5),'s',round(time.time()-t1,1),flush=True)
    del T,P,I,U,UI,Hx,HF,R,UE; gc.collect()
print('total',round(time.time()-t0,1))
